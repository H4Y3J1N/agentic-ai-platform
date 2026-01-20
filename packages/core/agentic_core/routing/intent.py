"""
Intent Types - Intent 분류 결과 타입 정의
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class IntentType(Enum):
    """Intent 유형"""
    # Tool 호출
    TOOL_CREATE = "tool_create"       # 생성 요청 (태스크, 문서 등)
    TOOL_SEARCH = "tool_search"       # 명시적 검색 요청
    TOOL_UPDATE = "tool_update"       # 수정 요청
    TOOL_DELETE = "tool_delete"       # 삭제 요청

    # RAG 기반 Q&A
    RAG_QA = "rag_qa"                 # 지식 기반 질의응답
    RAG_SUMMARY = "rag_summary"       # 요약 요청
    RAG_COMPARE = "rag_compare"       # 비교 분석 요청

    # 일반 대화
    CONVERSATION = "conversation"     # 인사, 잡담, 설명 요청
    CLARIFICATION = "clarification"   # 명확화 요청 ("뭐라고?", "다시 설명해줘")
    FEEDBACK = "feedback"             # 피드백 ("고마워", "잘했어")

    # 특수
    MULTI_INTENT = "multi_intent"     # 복합 인텐트
    UNKNOWN = "unknown"               # 분류 불가


class IntentConfidence(Enum):
    """분류 신뢰도"""
    HIGH = "high"           # 0.8 이상 - 바로 실행
    MEDIUM = "medium"       # 0.5-0.8 - 실행하되 확인 고려
    LOW = "low"             # 0.5 미만 - 사용자 확인 필요


@dataclass
class SubIntent:
    """복합 인텐트의 하위 인텐트"""
    type: IntentType
    query: str                          # 해당 인텐트에 대한 쿼리 부분
    tool_name: Optional[str] = None     # 사용할 Tool 이름
    params: Dict[str, Any] = field(default_factory=dict)
    order: int = 0                      # 실행 순서


@dataclass
class Intent:
    """
    Intent 분류 결과

    Attributes:
        type: 인텐트 유형
        confidence: 분류 신뢰도 (0.0 ~ 1.0)
        confidence_level: 신뢰도 레벨 (HIGH/MEDIUM/LOW)
        original_query: 원본 쿼리
        processed_query: 처리된 쿼리 (정규화, 키워드 추출 등)
        tool_name: 사용할 Tool 이름 (Tool 인텐트인 경우)
        params: Tool 파라미터 또는 추가 정보
        sub_intents: 복합 인텐트의 하위 인텐트 목록
        reasoning: LLM의 분류 이유 (디버깅용)
        metadata: 추가 메타데이터
    """
    type: IntentType
    confidence: float
    original_query: str
    processed_query: str = ""
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    sub_intents: List[SubIntent] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_level(self) -> IntentConfidence:
        """신뢰도 레벨 계산"""
        if self.confidence >= 0.8:
            return IntentConfidence.HIGH
        elif self.confidence >= 0.5:
            return IntentConfidence.MEDIUM
        else:
            return IntentConfidence.LOW

    @property
    def is_tool_intent(self) -> bool:
        """Tool 호출 인텐트인지"""
        return self.type in (
            IntentType.TOOL_CREATE,
            IntentType.TOOL_SEARCH,
            IntentType.TOOL_UPDATE,
            IntentType.TOOL_DELETE,
        )

    @property
    def is_rag_intent(self) -> bool:
        """RAG 기반 인텐트인지"""
        return self.type in (
            IntentType.RAG_QA,
            IntentType.RAG_SUMMARY,
            IntentType.RAG_COMPARE,
        )

    @property
    def is_conversation(self) -> bool:
        """일반 대화 인텐트인지"""
        return self.type in (
            IntentType.CONVERSATION,
            IntentType.CLARIFICATION,
            IntentType.FEEDBACK,
        )

    @property
    def needs_confirmation(self) -> bool:
        """사용자 확인이 필요한지"""
        # 낮은 신뢰도이거나 생성/수정/삭제 작업
        return (
            self.confidence_level == IntentConfidence.LOW
            or self.type in (IntentType.TOOL_CREATE, IntentType.TOOL_UPDATE, IntentType.TOOL_DELETE)
        )

    @property
    def is_multi_intent(self) -> bool:
        """복합 인텐트인지"""
        return self.type == IntentType.MULTI_INTENT or len(self.sub_intents) > 0

    def get_execution_order(self) -> List["Intent"]:
        """
        실행 순서대로 정렬된 인텐트 목록 반환

        복합 인텐트인 경우 sub_intents를 순서대로 반환
        """
        if not self.sub_intents:
            return [self]

        sorted_subs = sorted(self.sub_intents, key=lambda x: x.order)
        return [
            Intent(
                type=sub.type,
                confidence=self.confidence,
                original_query=sub.query,
                processed_query=sub.query,
                tool_name=sub.tool_name,
                params=sub.params,
            )
            for sub in sorted_subs
        ]

    def __repr__(self) -> str:
        return (
            f"Intent(type={self.type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"tool={self.tool_name})"
        )
