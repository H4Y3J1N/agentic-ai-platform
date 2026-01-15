"""
LoRA Loader

LoRA 어댑터 로딩/언로딩/스왑 관리
vLLM 서버 또는 로컬 모델과 연동
"""

from typing import Optional, Dict, List, Any, Set
from datetime import datetime
from collections import OrderedDict
import asyncio
import httpx

from .config import LoRAConfig, LoRAAdapter
from .registry import LoRAAdapterRegistry


class LRUCache:
    """LRU 캐시 (로드된 어댑터 관리용)"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: OrderedDict[str, datetime] = OrderedDict()

    def access(self, key: str) -> None:
        """키 접근 (순서 갱신)"""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = datetime.now()
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def get_lru(self) -> Optional[str]:
        """가장 오래된 키 반환"""
        if self._cache:
            return next(iter(self._cache))
        return None

    def remove(self, key: str) -> None:
        """키 제거"""
        self._cache.pop(key, None)

    def keys(self) -> List[str]:
        """모든 키"""
        return list(self._cache.keys())


class LoRALoader:
    """
    LoRA 로더

    어댑터 로딩/언로딩을 관리하고 vLLM 서버와 연동합니다.
    LRU 캐시를 사용하여 자주 사용하는 어댑터를 유지합니다.
    """

    def __init__(
        self,
        config: LoRAConfig,
        registry: LoRAAdapterRegistry,
    ):
        self.config = config
        self.registry = registry

        # 로드된 어댑터 추적
        self._loaded_adapters: Set[str] = set()

        # LRU 캐시
        self._lru_cache: Optional[LRUCache] = None
        if config.enable_lru_cache:
            self._lru_cache = LRUCache(config.lru_cache_size)

        # HTTP 클라이언트 (vLLM 연동용)
        self._http_client: Optional[httpx.AsyncClient] = None

        # 락 (동시 로딩 방지)
        self._lock = asyncio.Lock()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 lazy initialization"""
        if self._http_client is None and self.config.vllm_base_url:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.vllm_base_url,
                timeout=60.0,
            )
        if self._http_client is None:
            raise RuntimeError("vLLM base URL not configured")
        return self._http_client

    async def load_adapter(
        self,
        name: str,
        force: bool = False
    ) -> bool:
        """
        어댑터 로드

        Args:
            name: 어댑터 이름
            force: 이미 로드된 경우에도 강제 로드

        Returns:
            성공 여부
        """
        async with self._lock:
            adapter = self.registry.get(name)
            if not adapter:
                raise ValueError(f"Adapter '{name}' not found in registry")

            # 이미 로드됨
            if name in self._loaded_adapters and not force:
                if self._lru_cache:
                    self._lru_cache.access(name)
                return True

            # 최대 로드 수 확인
            if len(self._loaded_adapters) >= self.config.max_loaded_adapters:
                # LRU 어댑터 언로드
                await self._evict_lru_adapter()

            # vLLM 서버에 로드 요청
            success = await self._load_to_vllm(adapter)

            if success:
                self._loaded_adapters.add(name)
                self.registry.update_status(name, True, datetime.now())

                if self._lru_cache:
                    self._lru_cache.access(name)

            return success

    async def unload_adapter(self, name: str) -> bool:
        """
        어댑터 언로드

        Args:
            name: 어댑터 이름

        Returns:
            성공 여부
        """
        async with self._lock:
            if name not in self._loaded_adapters:
                return True  # 이미 언로드됨

            success = await self._unload_from_vllm(name)

            if success:
                self._loaded_adapters.discard(name)
                self.registry.update_status(name, False)

                if self._lru_cache:
                    self._lru_cache.remove(name)

            return success

    async def ensure_loaded(self, name: str) -> bool:
        """
        어댑터가 로드되어 있는지 확인하고, 없으면 로드

        Args:
            name: 어댑터 이름

        Returns:
            성공 여부
        """
        if name in self._loaded_adapters:
            if self._lru_cache:
                self._lru_cache.access(name)
            return True
        return await self.load_adapter(name)

    async def get_for_tenant(
        self,
        tenant_id: str,
        conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        테넌트에 맞는 어댑터를 로드하고 이름 반환

        Args:
            tenant_id: 테넌트 ID
            conditions: 추가 조건

        Returns:
            어댑터 이름 또는 None
        """
        adapter = self.registry.get_adapter_for_tenant(tenant_id, conditions)
        if not adapter:
            return None

        success = await self.ensure_loaded(adapter.name)
        return adapter.name if success else None

    def get_loaded_adapters(self) -> List[str]:
        """로드된 어댑터 목록"""
        return list(self._loaded_adapters)

    async def _evict_lru_adapter(self) -> None:
        """LRU 어댑터 언로드"""
        if self._lru_cache:
            lru_name = self._lru_cache.get_lru()
            if lru_name:
                await self._unload_from_vllm(lru_name)
                self._loaded_adapters.discard(lru_name)
                self.registry.update_status(lru_name, False)
                self._lru_cache.remove(lru_name)
        elif self._loaded_adapters:
            # LRU 캐시 없으면 아무거나 언로드
            name = next(iter(self._loaded_adapters))
            await self._unload_from_vllm(name)
            self._loaded_adapters.discard(name)
            self.registry.update_status(name, False)

    async def _load_to_vllm(self, adapter: LoRAAdapter) -> bool:
        """
        vLLM 서버에 어댑터 로드 요청

        vLLM은 동적 LoRA 로딩을 지원합니다.
        POST /v1/load_lora_adapter
        """
        if not self.config.vllm_base_url:
            # vLLM 연동 없이 레지스트리만 사용
            return True

        try:
            client = await self._get_http_client()
            response = await client.post(
                "/v1/load_lora_adapter",
                json={
                    "lora_name": adapter.name,
                    "lora_path": adapter.path,
                }
            )
            return response.status_code == 200
        except httpx.HTTPError:
            # vLLM이 동적 로딩을 지원하지 않는 경우
            # 서버 시작 시 미리 로드되어 있다고 가정
            return True

    async def _unload_from_vllm(self, name: str) -> bool:
        """
        vLLM 서버에서 어댑터 언로드 요청

        POST /v1/unload_lora_adapter
        """
        if not self.config.vllm_base_url:
            return True

        try:
            client = await self._get_http_client()
            response = await client.post(
                "/v1/unload_lora_adapter",
                json={"lora_name": name}
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return True

    async def sync_with_vllm(self) -> Dict[str, Any]:
        """
        vLLM 서버와 상태 동기화

        서버에 로드된 어댑터 목록을 조회하고 레지스트리와 동기화
        """
        if not self.config.vllm_base_url:
            return {"synced": False, "reason": "vLLM not configured"}

        try:
            client = await self._get_http_client()
            response = await client.get("/v1/models")

            if response.status_code == 200:
                data = response.json()
                loaded_on_server = set()

                # vLLM은 모델 목록에 LoRA 어댑터도 포함
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # 형식: "base-model:lora-adapter"
                    if ":" in model_id:
                        _, adapter_name = model_id.split(":", 1)
                        loaded_on_server.add(adapter_name)

                # 레지스트리 상태 업데이트
                for name in loaded_on_server:
                    if name in self._loaded_adapters:
                        continue
                    adapter = self.registry.get(name)
                    if adapter:
                        self._loaded_adapters.add(name)
                        self.registry.update_status(name, True, datetime.now())

                # 서버에 없는 어댑터 상태 업데이트
                for name in list(self._loaded_adapters):
                    if name not in loaded_on_server:
                        self._loaded_adapters.discard(name)
                        self.registry.update_status(name, False)

                return {
                    "synced": True,
                    "loaded_count": len(self._loaded_adapters),
                    "loaded_adapters": list(self._loaded_adapters),
                }

        except httpx.HTTPError as e:
            return {"synced": False, "error": str(e)}

        return {"synced": False, "reason": "Unknown error"}

    async def close(self) -> None:
        """리소스 정리"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
