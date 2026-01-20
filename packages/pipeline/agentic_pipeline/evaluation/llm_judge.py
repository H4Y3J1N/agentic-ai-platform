"""
LLM Judge

LiteLLM 기반 LLM-as-a-Judge 구현
평가 메트릭에서 사용하는 공통 LLM 호출 래퍼
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeConfig:
    """LLM Judge 설정"""
    model: str = "gemini/gemini-1.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0


class LLMJudge:
    """
    LLM-as-a-Judge

    평가 프롬프트를 LLM에 보내고 응답을 받는 래퍼 클래스.
    RAGEvaluator, LLMEvaluator의 llm_judge 파라미터로 사용.
    """

    def __init__(self, config: Optional[LLMJudgeConfig] = None):
        self.config = config or LLMJudgeConfig()
        self._acompletion = None

    async def _ensure_llm(self):
        """LLM 클라이언트 초기화"""
        if self._acompletion is None:
            try:
                from litellm import acompletion
                self._acompletion = acompletion
            except ImportError:
                raise ImportError(
                    "litellm is required for LLMJudge. "
                    "Install with: pip install litellm"
                )

    async def __call__(self, prompt: str) -> str:
        """
        LLM 호출

        Args:
            prompt: 평가 프롬프트

        Returns:
            LLM 응답 텍스트
        """
        await self._ensure_llm()

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._acompletion(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}): {e}, retrying..."
                    )
                    await asyncio.sleep(self.config.retry_delay)

        raise RuntimeError(f"LLM call failed after {self.config.max_retries + 1} attempts: {last_error}")

    def as_callable(self) -> Callable:
        """
        콜백 함수로 변환

        Returns:
            async callable for llm_judge parameter
        """
        return self.__call__


def create_llm_judge(
    model: str = "gemini/gemini-1.5-flash",
    temperature: float = 0.0,
    **kwargs
) -> LLMJudge:
    """
    LLM Judge 팩토리 함수

    Args:
        model: LLM 모델 (LiteLLM 형식)
        temperature: 생성 온도 (평가용이므로 0 권장)
        **kwargs: 추가 설정

    Returns:
        LLMJudge 인스턴스

    Example:
        >>> judge = create_llm_judge(model="gemini/gemini-1.5-flash")
        >>> evaluator = RAGEvaluator(llm_judge=judge)
    """
    config = LLMJudgeConfig(
        model=model,
        temperature=temperature,
        **{k: v for k, v in kwargs.items() if k in LLMJudgeConfig.__dataclass_fields__}
    )
    return LLMJudge(config)
