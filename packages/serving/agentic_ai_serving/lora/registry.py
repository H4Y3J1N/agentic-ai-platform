"""
LoRA Adapter Registry

어댑터 등록, 조회, 삭제 및 테넌트 매핑 관리
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import json
from pathlib import Path

from .config import LoRAConfig, LoRAAdapter, TenantLoRAMapping


class LoRAAdapterRegistry:
    """
    LoRA 어댑터 레지스트리

    어댑터 메타데이터 관리 및 테넌트 라우팅을 담당합니다.
    실제 모델 로딩은 LoRALoader가 담당합니다.
    """

    def __init__(self, config: LoRAConfig):
        self.config = config
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._tenant_mappings: Dict[str, TenantLoRAMapping] = {}

        # 설정에서 초기 데이터 로드
        for adapter in config.adapters:
            self._adapters[adapter.name] = adapter
        for mapping in config.tenant_mappings:
            self._tenant_mappings[mapping.tenant_id] = mapping

    # ==================== 어댑터 관리 ====================

    def register(self, adapter: LoRAAdapter) -> None:
        """
        어댑터 등록

        Args:
            adapter: 등록할 LoRA 어댑터
        """
        if adapter.name in self._adapters:
            raise ValueError(f"Adapter '{adapter.name}' already exists")
        self._adapters[adapter.name] = adapter

    def unregister(self, name: str) -> Optional[LoRAAdapter]:
        """
        어댑터 등록 해제

        Args:
            name: 어댑터 이름

        Returns:
            제거된 어댑터 또는 None
        """
        return self._adapters.pop(name, None)

    def get(self, name: str) -> Optional[LoRAAdapter]:
        """
        어댑터 조회

        Args:
            name: 어댑터 이름

        Returns:
            어댑터 또는 None
        """
        return self._adapters.get(name)

    def list_adapters(
        self,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        loaded_only: bool = False,
    ) -> List[LoRAAdapter]:
        """
        어댑터 목록 조회

        Args:
            base_model: 베이스 모델로 필터링
            tags: 태그로 필터링 (AND 조건)
            loaded_only: 로드된 어댑터만

        Returns:
            어댑터 리스트
        """
        adapters = list(self._adapters.values())

        if base_model:
            adapters = [a for a in adapters if a.base_model == base_model]

        if tags:
            adapters = [a for a in adapters if all(t in a.tags for t in tags)]

        if loaded_only:
            adapters = [a for a in adapters if a.is_loaded]

        return adapters

    def update_status(
        self,
        name: str,
        is_loaded: bool,
        loaded_at: Optional[datetime] = None
    ) -> None:
        """
        어댑터 로드 상태 업데이트

        Args:
            name: 어댑터 이름
            is_loaded: 로드 상태
            loaded_at: 로드 시간
        """
        adapter = self._adapters.get(name)
        if adapter:
            adapter.is_loaded = is_loaded
            adapter.loaded_at = loaded_at if is_loaded else None

    # ==================== 테넌트 매핑 ====================

    def set_tenant_mapping(self, mapping: TenantLoRAMapping) -> None:
        """
        테넌트-어댑터 매핑 설정

        Args:
            mapping: 테넌트 매핑
        """
        # 어댑터 존재 확인
        if mapping.adapter_name not in self._adapters:
            raise ValueError(f"Adapter '{mapping.adapter_name}' not found")

        mapping.updated_at = datetime.now()
        self._tenant_mappings[mapping.tenant_id] = mapping

    def remove_tenant_mapping(self, tenant_id: str) -> Optional[TenantLoRAMapping]:
        """
        테넌트 매핑 제거

        Args:
            tenant_id: 테넌트 ID

        Returns:
            제거된 매핑 또는 None
        """
        return self._tenant_mappings.pop(tenant_id, None)

    def get_adapter_for_tenant(
        self,
        tenant_id: str,
        conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[LoRAAdapter]:
        """
        테넌트에 매핑된 어댑터 조회

        Args:
            tenant_id: 테넌트 ID
            conditions: 추가 매핑 조건

        Returns:
            매핑된 어댑터 또는 기본 어댑터
        """
        mapping = self._tenant_mappings.get(tenant_id)

        if mapping and mapping.enabled:
            # 조건 매칭 (있는 경우)
            if conditions and mapping.conditions:
                if not self._match_conditions(mapping.conditions, conditions):
                    mapping = None

        if mapping:
            return self._adapters.get(mapping.adapter_name)

        # 기본 어댑터 반환
        if self.config.default_adapter:
            return self._adapters.get(self.config.default_adapter)

        return None

    def list_tenant_mappings(self) -> List[TenantLoRAMapping]:
        """모든 테넌트 매핑 조회"""
        return list(self._tenant_mappings.values())

    def _match_conditions(
        self,
        mapping_conditions: Dict[str, Any],
        request_conditions: Dict[str, Any]
    ) -> bool:
        """조건 매칭 확인"""
        for key, value in mapping_conditions.items():
            if key not in request_conditions:
                return False
            if request_conditions[key] != value:
                return False
        return True

    # ==================== 영속화 ====================

    def save_to_file(self, path: str) -> None:
        """
        레지스트리 상태를 파일로 저장

        Args:
            path: 저장 경로
        """
        data = {
            "adapters": [a.model_dump() for a in self._adapters.values()],
            "tenant_mappings": [m.model_dump() for m in self._tenant_mappings.values()],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_file(self, path: str) -> None:
        """
        파일에서 레지스트리 상태 로드

        Args:
            path: 파일 경로
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._adapters.clear()
        self._tenant_mappings.clear()

        for adapter_data in data.get("adapters", []):
            adapter = LoRAAdapter(**adapter_data)
            adapter.is_loaded = False  # 로드 상태 초기화
            self._adapters[adapter.name] = adapter

        for mapping_data in data.get("tenant_mappings", []):
            mapping = TenantLoRAMapping(**mapping_data)
            self._tenant_mappings[mapping.tenant_id] = mapping

    # ==================== 유틸리티 ====================

    def get_stats(self) -> Dict[str, Any]:
        """레지스트리 통계"""
        loaded_adapters = [a for a in self._adapters.values() if a.is_loaded]
        active_mappings = [m for m in self._tenant_mappings.values() if m.enabled]

        return {
            "total_adapters": len(self._adapters),
            "loaded_adapters": len(loaded_adapters),
            "total_tenant_mappings": len(self._tenant_mappings),
            "active_tenant_mappings": len(active_mappings),
            "adapters_by_base_model": self._count_by_base_model(),
        }

    def _count_by_base_model(self) -> Dict[str, int]:
        """베이스 모델별 어댑터 수"""
        counts: Dict[str, int] = {}
        for adapter in self._adapters.values():
            counts[adapter.base_model] = counts.get(adapter.base_model, 0) + 1
        return counts
