"""
vLLM Configuration

vLLM 서버 및 모델 설정
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class DType(str, Enum):
    """데이터 타입"""
    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class VLLMModelConfig(BaseModel):
    """vLLM 모델 설정"""

    model_name: str = Field(..., description="HuggingFace 모델 이름 또는 로컬 경로")

    # 텐서 병렬화
    tensor_parallel_size: int = Field(default=1, description="GPU 텐서 병렬화 수")
    pipeline_parallel_size: int = Field(default=1, description="파이프라인 병렬화 수")

    # 메모리 설정
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="GPU 메모리 사용률 (0.0 ~ 1.0)"
    )
    max_model_len: Optional[int] = Field(
        default=None,
        description="최대 컨텍스트 길이 (None이면 모델 기본값)"
    )

    # 데이터 타입
    dtype: DType = Field(default=DType.AUTO, description="모델 데이터 타입")

    # 양자화
    quantization: Optional[str] = Field(
        default=None,
        description="양자화 방법 (awq, gptq, squeezellm, fp8 등)"
    )

    # LoRA 지원
    enable_lora: bool = Field(default=False, description="LoRA 어댑터 지원 활성화")
    max_loras: int = Field(default=1, description="동시 로드 가능한 최대 LoRA 수")
    max_lora_rank: int = Field(default=64, description="최대 LoRA rank")
    lora_extra_vocab_size: int = Field(default=256, description="LoRA 추가 vocabulary 크기")

    # 추가 설정
    trust_remote_code: bool = Field(default=False, description="원격 코드 신뢰 여부")
    revision: Optional[str] = Field(default=None, description="모델 revision/branch")
    tokenizer: Optional[str] = Field(default=None, description="별도 토크나이저 경로")

    extra_args: Dict[str, Any] = Field(default_factory=dict, description="추가 vLLM 인자")


class VLLMConfig(BaseModel):
    """vLLM 서버 연결 설정"""

    # 서버 연결
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM 서버 URL (OpenAI-compatible API)"
    )
    api_key: str = Field(
        default="EMPTY",
        description="API 키 (vLLM은 기본적으로 인증 불필요)"
    )

    # 타임아웃
    timeout: float = Field(default=60.0, description="요청 타임아웃 (초)")
    max_retries: int = Field(default=3, description="최대 재시도 횟수")

    # 기본 생성 설정
    default_model: Optional[str] = Field(
        default=None,
        description="기본 모델 이름 (서버에 로드된 모델)"
    )
    default_max_tokens: int = Field(default=1024, description="기본 최대 토큰 수")
    default_temperature: float = Field(default=0.7, description="기본 temperature")

    # LoRA 관련
    enable_lora_routing: bool = Field(
        default=False,
        description="테넌트별 LoRA 라우팅 활성화"
    )

    # 모델 설정 (서버 시작 시 사용)
    model_config: Optional[VLLMModelConfig] = Field(
        default=None,
        description="모델 설정 (서버 시작 시)"
    )

    # 헬스체크
    health_check_interval: float = Field(
        default=30.0,
        description="헬스체크 간격 (초)"
    )

    model_config_dict = {"protected_namespaces": ()}
