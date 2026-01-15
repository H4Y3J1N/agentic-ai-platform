"""
LoRA Configuration

LoRA 어댑터 및 테넌트 매핑 설정
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class LoRATargetModules(str, Enum):
    """일반적인 LoRA 타겟 모듈 프리셋"""
    LLAMA = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    MISTRAL = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    GPT = "c_attn,c_proj,c_fc"
    CUSTOM = "custom"


class LoRAAdapter(BaseModel):
    """LoRA 어댑터 정보"""

    name: str = Field(..., description="어댑터 고유 이름")
    path: str = Field(..., description="어댑터 파일 경로 (로컬 또는 HuggingFace)")

    # 어댑터 메타데이터
    base_model: str = Field(..., description="베이스 모델 이름")
    description: Optional[str] = Field(default=None, description="어댑터 설명")
    version: str = Field(default="1.0.0", description="어댑터 버전")

    # LoRA 파라미터 (정보용)
    rank: int = Field(default=8, description="LoRA rank (r)")
    alpha: int = Field(default=16, description="LoRA alpha")
    dropout: float = Field(default=0.0, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="타겟 모듈 리스트"
    )

    # 상태
    is_loaded: bool = Field(default=False, description="현재 로드 상태")
    loaded_at: Optional[datetime] = Field(default=None, description="로드 시간")

    # 추가 메타데이터
    tags: List[str] = Field(default_factory=list, description="태그")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

    model_config = {"protected_namespaces": ()}


class TenantLoRAMapping(BaseModel):
    """테넌트별 LoRA 매핑"""

    tenant_id: str = Field(..., description="테넌트 ID")
    adapter_name: str = Field(..., description="사용할 LoRA 어댑터 이름")

    # 매핑 조건
    priority: int = Field(default=0, description="우선순위 (높을수록 우선)")
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="추가 매핑 조건 (예: use_case, language 등)"
    )

    # 상태
    enabled: bool = Field(default=True, description="활성화 여부")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class LoRAConfig(BaseModel):
    """LoRA 관리 설정"""

    # 어댑터 저장 경로
    adapters_base_path: str = Field(
        default="./lora_adapters",
        description="LoRA 어댑터 기본 저장 경로"
    )

    # 로딩 설정
    max_loaded_adapters: int = Field(
        default=4,
        description="동시에 로드할 수 있는 최대 어댑터 수"
    )
    auto_unload_inactive: bool = Field(
        default=True,
        description="비활성 어댑터 자동 언로드"
    )
    inactive_timeout_seconds: int = Field(
        default=300,
        description="비활성 어댑터 언로드 타임아웃 (초)"
    )

    # LRU 캐시 설정
    enable_lru_cache: bool = Field(
        default=True,
        description="LRU 캐시 활성화 (자주 사용하는 어댑터 유지)"
    )
    lru_cache_size: int = Field(
        default=4,
        description="LRU 캐시 크기"
    )

    # vLLM 연동 설정
    vllm_base_url: Optional[str] = Field(
        default=None,
        description="vLLM 서버 URL (동적 LoRA 로딩 시)"
    )

    # 테넌트 매핑
    enable_tenant_routing: bool = Field(
        default=False,
        description="테넌트별 LoRA 라우팅 활성화"
    )
    default_adapter: Optional[str] = Field(
        default=None,
        description="매핑되지 않은 테넌트의 기본 어댑터"
    )

    # 등록된 어댑터 목록
    adapters: List[LoRAAdapter] = Field(
        default_factory=list,
        description="등록된 어댑터 목록"
    )

    # 테넌트 매핑 목록
    tenant_mappings: List[TenantLoRAMapping] = Field(
        default_factory=list,
        description="테넌트-어댑터 매핑 목록"
    )
