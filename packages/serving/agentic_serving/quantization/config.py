"""
Quantization Configuration

다양한 양자화 방법에 대한 설정 스키마
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class QuantizationMethod(str, Enum):
    """양자화 방법"""
    NONE = "none"
    QLORA = "qlora"      # 4-bit QLoRA (bitsandbytes)
    EXL2 = "exl2"        # ExLlamaV2
    GPTQ = "gptq"        # GPT-Q
    AWQ = "awq"          # Activation-aware Weight Quantization
    FP8 = "fp8"          # FP8 quantization (vLLM native)
    GGUF = "gguf"        # llama.cpp format


class ComputeDType(str, Enum):
    """연산 데이터 타입"""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class QLoRAConfig(BaseModel):
    """
    QLoRA 설정 (bitsandbytes 4-bit)

    QLoRA는 양자화된 베이스 모델 위에 LoRA를 학습/추론합니다.
    """

    # 양자화 비트
    bits: int = Field(default=4, description="양자화 비트 수 (4 또는 8)")

    # bitsandbytes 설정
    bnb_4bit_compute_dtype: ComputeDType = Field(
        default=ComputeDType.BFLOAT16,
        description="연산 데이터 타입"
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="양자화 타입 (nf4 또는 fp4)"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="이중 양자화 사용 (메모리 추가 절약)"
    )

    # LoRA 설정 (학습 시)
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")
    lora_target_modules: Optional[List[str]] = Field(
        default=None,
        description="타겟 모듈 (None이면 자동 감지)"
    )

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, v):
        if v not in [4, 8]:
            raise ValueError("bits must be 4 or 8")
        return v


class EXL2Config(BaseModel):
    """
    EXL2 설정 (ExLlamaV2)

    EXL2는 가변 비트율 양자화를 지원하며 매우 빠른 추론이 가능합니다.
    """

    # 비트율 (bpw: bits per weight)
    bits_per_weight: float = Field(
        default=4.0,
        ge=2.0,
        le=8.0,
        description="가중치당 비트 수 (2.0 ~ 8.0)"
    )

    # 모델 경로
    model_path: str = Field(..., description="EXL2 양자화된 모델 경로")

    # 캐시 설정
    cache_mode: str = Field(
        default="FP16",
        description="KV 캐시 모드 (FP16, FP8, Q8, Q4)"
    )
    cache_size: Optional[int] = Field(
        default=None,
        description="캐시 크기 (토큰 수, None이면 자동)"
    )

    # GPU 설정
    gpu_split: Optional[List[float]] = Field(
        default=None,
        description="멀티 GPU 분할 비율"
    )

    # 추론 설정
    max_seq_len: Optional[int] = Field(
        default=None,
        description="최대 시퀀스 길이"
    )
    rope_scale: float = Field(default=1.0, description="RoPE 스케일")
    rope_alpha: float = Field(default=1.0, description="RoPE alpha (NTK)")


class GPTQConfig(BaseModel):
    """
    GPTQ 설정

    GPTQ는 사후 학습 양자화 방법으로 널리 사용됩니다.
    """

    # 양자화 비트
    bits: int = Field(default=4, description="양자화 비트 수 (2, 3, 4, 8)")

    # 그룹 크기
    group_size: int = Field(
        default=128,
        description="양자화 그룹 크기 (-1이면 전체)"
    )

    # 활성화 순서
    desc_act: bool = Field(
        default=True,
        description="활성화 기반 열 순서 지정"
    )

    # 모델 경로
    model_path: Optional[str] = Field(
        default=None,
        description="GPTQ 모델 경로 (None이면 HuggingFace에서 로드)"
    )

    # Marlin 커널 사용 (더 빠름)
    use_marlin: bool = Field(
        default=False,
        description="Marlin 커널 사용 (vLLM에서 지원)"
    )

    @field_validator("bits")
    @classmethod
    def validate_bits(cls, v):
        if v not in [2, 3, 4, 8]:
            raise ValueError("bits must be 2, 3, 4, or 8")
        return v


class AWQConfig(BaseModel):
    """
    AWQ 설정 (Activation-aware Weight Quantization)

    AWQ는 활성화 분포를 고려하여 중요한 가중치를 보존합니다.
    """

    # 양자화 비트
    bits: int = Field(default=4, description="양자화 비트 수")

    # 그룹 크기
    group_size: int = Field(
        default=128,
        description="양자화 그룹 크기"
    )

    # 제로 포인트
    zero_point: bool = Field(
        default=True,
        description="제로 포인트 사용"
    )

    # 모델 경로
    model_path: Optional[str] = Field(
        default=None,
        description="AWQ 모델 경로"
    )

    # 버전
    version: str = Field(
        default="gemm",
        description="AWQ 버전 (gemm, gemv)"
    )


class QuantizationConfig(BaseModel):
    """
    통합 양자화 설정

    다양한 양자화 방법을 하나의 설정으로 관리합니다.
    """

    # 양자화 방법
    method: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description="양자화 방법"
    )

    # 각 방법별 설정
    qlora: Optional[QLoRAConfig] = Field(default=None)
    exl2: Optional[EXL2Config] = Field(default=None)
    gptq: Optional[GPTQConfig] = Field(default=None)
    awq: Optional[AWQConfig] = Field(default=None)

    # 공통 설정
    device_map: str = Field(
        default="auto",
        description="디바이스 매핑 (auto, cuda, cpu)"
    )
    max_memory: Optional[Dict[str, str]] = Field(
        default=None,
        description="GPU별 최대 메모리 (예: {'0': '20GB', 'cpu': '30GB'})"
    )

    # vLLM 설정 (vLLM 사용 시)
    vllm_quantization: Optional[str] = Field(
        default=None,
        description="vLLM 양자화 옵션 (awq, gptq, squeezellm, fp8)"
    )

    def get_active_config(self) -> Optional[BaseModel]:
        """활성화된 양자화 설정 반환"""
        config_map = {
            QuantizationMethod.QLORA: self.qlora,
            QuantizationMethod.EXL2: self.exl2,
            QuantizationMethod.GPTQ: self.gptq,
            QuantizationMethod.AWQ: self.awq,
        }
        return config_map.get(self.method)

    def to_transformers_kwargs(self) -> Dict[str, Any]:
        """transformers 라이브러리용 kwargs 생성"""
        if self.method == QuantizationMethod.NONE:
            return {}

        if self.method == QuantizationMethod.QLORA and self.qlora:
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=self.qlora.bits == 4,
                    load_in_8bit=self.qlora.bits == 8,
                    bnb_4bit_compute_dtype=self.qlora.bnb_4bit_compute_dtype.value,
                    bnb_4bit_quant_type=self.qlora.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.qlora.bnb_4bit_use_double_quant,
                )
            }

        if self.method == QuantizationMethod.GPTQ and self.gptq:
            from transformers import GPTQConfig as HFGPTQConfig
            return {
                "quantization_config": HFGPTQConfig(
                    bits=self.gptq.bits,
                    group_size=self.gptq.group_size,
                    desc_act=self.gptq.desc_act,
                )
            }

        if self.method == QuantizationMethod.AWQ and self.awq:
            from transformers import AwqConfig as HFAwqConfig
            return {
                "quantization_config": HFAwqConfig(
                    bits=self.awq.bits,
                    group_size=self.awq.group_size,
                    zero_point=self.awq.zero_point,
                    version=self.awq.version,
                )
            }

        return {}

    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """vLLM용 kwargs 생성"""
        if self.vllm_quantization:
            return {"quantization": self.vllm_quantization}

        method_to_vllm = {
            QuantizationMethod.AWQ: "awq",
            QuantizationMethod.GPTQ: "gptq",
            QuantizationMethod.FP8: "fp8",
        }

        if self.method in method_to_vllm:
            return {"quantization": method_to_vllm[self.method]}

        return {}
