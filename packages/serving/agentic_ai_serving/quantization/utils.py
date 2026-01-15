"""
Quantization Utilities

양자화 관련 유틸리티 함수
- 메모리 사용량 추정
- 최적 양자화 방법 추천
- 설정 검증
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .config import QuantizationMethod, QuantizationConfig


@dataclass
class MemoryEstimate:
    """메모리 사용량 추정"""
    model_memory_gb: float
    kv_cache_memory_gb: float
    overhead_memory_gb: float
    total_memory_gb: float
    fits_in_vram: bool
    recommended_gpu: str


# 알려진 모델 파라미터 수 (billions)
KNOWN_MODEL_PARAMS = {
    "llama-7b": 7.0,
    "llama-13b": 13.0,
    "llama-70b": 70.0,
    "mistral-7b": 7.0,
    "mixtral-8x7b": 46.7,  # MoE
    "qwen-7b": 7.0,
    "qwen-14b": 14.0,
    "qwen-72b": 72.0,
}

# GPU VRAM 용량 (GB)
GPU_VRAM = {
    "rtx-3090": 24,
    "rtx-4090": 24,
    "a100-40gb": 40,
    "a100-80gb": 80,
    "h100": 80,
    "a6000": 48,
    "rtx-3080": 10,
    "rtx-4080": 16,
}


def estimate_memory_usage(
    model_params_billions: float,
    quantization_method: QuantizationMethod,
    context_length: int = 4096,
    batch_size: int = 1,
    bits: int = 4,
) -> MemoryEstimate:
    """
    모델 메모리 사용량 추정

    Args:
        model_params_billions: 모델 파라미터 수 (십억 단위)
        quantization_method: 양자화 방법
        context_length: 컨텍스트 길이
        batch_size: 배치 크기
        bits: 양자화 비트 수

    Returns:
        MemoryEstimate: 메모리 추정치
    """
    # 양자화별 비트/파라미터
    bits_per_param = {
        QuantizationMethod.NONE: 16,      # FP16
        QuantizationMethod.QLORA: bits,
        QuantizationMethod.EXL2: bits,
        QuantizationMethod.GPTQ: bits,
        QuantizationMethod.AWQ: bits,
        QuantizationMethod.FP8: 8,
        QuantizationMethod.GGUF: bits,
    }

    effective_bits = bits_per_param.get(quantization_method, 16)

    # 모델 가중치 메모리 (GB)
    model_memory_gb = (model_params_billions * 1e9 * effective_bits) / (8 * 1e9)

    # KV 캐시 메모리 추정 (FP16 가정)
    # 대략적인 추정: 레이어 수 * 2 * hidden_size * context_length * batch_size * 2 bytes
    # 간단한 휴리스틱: params * 0.05 * context_length / 4096
    kv_cache_factor = context_length / 4096
    kv_cache_memory_gb = model_params_billions * 0.1 * kv_cache_factor * batch_size

    # 오버헤드 (활성화, CUDA 컨텍스트 등)
    overhead_memory_gb = max(2.0, model_memory_gb * 0.1)

    total_memory_gb = model_memory_gb + kv_cache_memory_gb + overhead_memory_gb

    # 추천 GPU 결정
    recommended_gpu = "a100-80gb"  # 기본값
    for gpu, vram in sorted(GPU_VRAM.items(), key=lambda x: x[1]):
        if vram >= total_memory_gb * 1.1:  # 10% 여유
            recommended_gpu = gpu
            break

    fits_in_vram = total_memory_gb < 24  # RTX 3090/4090 기준

    return MemoryEstimate(
        model_memory_gb=round(model_memory_gb, 2),
        kv_cache_memory_gb=round(kv_cache_memory_gb, 2),
        overhead_memory_gb=round(overhead_memory_gb, 2),
        total_memory_gb=round(total_memory_gb, 2),
        fits_in_vram=fits_in_vram,
        recommended_gpu=recommended_gpu,
    )


def get_recommended_quantization(
    model_params_billions: float,
    available_vram_gb: float,
    use_case: str = "inference",
    priority: str = "balanced",
) -> Tuple[QuantizationMethod, int, Dict[str, Any]]:
    """
    최적 양자화 방법 추천

    Args:
        model_params_billions: 모델 파라미터 수
        available_vram_gb: 사용 가능한 VRAM (GB)
        use_case: 사용 목적 ("inference", "training", "finetuning")
        priority: 우선순위 ("speed", "quality", "balanced")

    Returns:
        Tuple[QuantizationMethod, bits, additional_config]: 추천 설정
    """
    # FP16 메모리 추정
    fp16_memory = model_params_billions * 2  # 대략 2GB per billion params

    # VRAM에 여유 있으면 양자화 불필요
    if fp16_memory * 1.3 < available_vram_gb:
        return QuantizationMethod.NONE, 16, {}

    # 사용 목적별 추천
    if use_case == "finetuning":
        # 파인튜닝은 QLoRA 권장
        bits = 4 if fp16_memory > available_vram_gb * 2 else 8
        return QuantizationMethod.QLORA, bits, {
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

    if use_case == "inference":
        # 추론은 속도/품질 트레이드오프
        if priority == "speed":
            # EXL2가 가장 빠름
            bits = 4.0 if fp16_memory > available_vram_gb * 3 else 5.0
            return QuantizationMethod.EXL2, int(bits), {"bits_per_weight": bits}

        if priority == "quality":
            # AWQ가 품질이 좋음
            bits = 4
            return QuantizationMethod.AWQ, bits, {"group_size": 128}

        # balanced: GPTQ (널리 호환)
        bits = 4
        return QuantizationMethod.GPTQ, bits, {"group_size": 128, "desc_act": True}

    # 기본: 4비트 GPTQ
    return QuantizationMethod.GPTQ, 4, {}


def validate_quantization_config(
    config: QuantizationConfig,
    model_params_billions: Optional[float] = None,
    available_vram_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    양자화 설정 검증

    Args:
        config: 양자화 설정
        model_params_billions: 모델 파라미터 수 (검증용)
        available_vram_gb: 사용 가능한 VRAM (검증용)

    Returns:
        검증 결과 딕셔너리
    """
    result = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # 방법별 설정 존재 확인
    if config.method == QuantizationMethod.QLORA and not config.qlora:
        result["errors"].append("QLoRA method selected but qlora config is missing")
        result["valid"] = False

    if config.method == QuantizationMethod.EXL2 and not config.exl2:
        result["errors"].append("EXL2 method selected but exl2 config is missing")
        result["valid"] = False

    if config.method == QuantizationMethod.GPTQ and not config.gptq:
        result["errors"].append("GPTQ method selected but gptq config is missing")
        result["valid"] = False

    if config.method == QuantizationMethod.AWQ and not config.awq:
        result["errors"].append("AWQ method selected but awq config is missing")
        result["valid"] = False

    # 메모리 검증
    if model_params_billions and available_vram_gb:
        active_config = config.get_active_config()
        bits = 4
        if hasattr(active_config, "bits"):
            bits = active_config.bits
        elif hasattr(active_config, "bits_per_weight"):
            bits = int(active_config.bits_per_weight)

        estimate = estimate_memory_usage(
            model_params_billions,
            config.method,
            bits=bits,
        )

        if estimate.total_memory_gb > available_vram_gb:
            result["warnings"].append(
                f"Estimated memory ({estimate.total_memory_gb}GB) exceeds "
                f"available VRAM ({available_vram_gb}GB)"
            )
            result["recommendations"].append(
                f"Consider using lower bits or {estimate.recommended_gpu}"
            )

    # EXL2 특수 검증
    if config.method == QuantizationMethod.EXL2 and config.exl2:
        if config.exl2.bits_per_weight < 3.0:
            result["warnings"].append(
                "Very low bits_per_weight (<3.0) may significantly degrade quality"
            )

    # QLoRA + vLLM 경고
    if config.method == QuantizationMethod.QLORA and config.vllm_quantization:
        result["warnings"].append(
            "QLoRA is not natively supported by vLLM. "
            "Consider using AWQ or GPTQ for vLLM serving."
        )

    return result


def get_model_params(model_name: str) -> Optional[float]:
    """
    알려진 모델의 파라미터 수 반환

    Args:
        model_name: 모델 이름

    Returns:
        파라미터 수 (billions) 또는 None
    """
    model_name_lower = model_name.lower()

    for known_name, params in KNOWN_MODEL_PARAMS.items():
        if known_name in model_name_lower:
            return params

    # 이름에서 추론 시도
    import re
    match = re.search(r"(\d+)b", model_name_lower)
    if match:
        return float(match.group(1))

    return None
