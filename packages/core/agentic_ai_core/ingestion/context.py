"""
Pipeline Context

파이프라인 스테이지 간 데이터 전달을 위한 컨텍스트 클래스
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from ..schema import (
    Document,
    DocumentType,
    SourceType,
    ParsedContent,
    Entity,
    Relationship,
    DecisionMapping,
    Chunk,
)


class ProcessingStatus(str, Enum):
    """처리 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class SourceItem:
    """원본 데이터 아이템"""
    id: str
    source_type: SourceType
    raw_data: Dict[str, Any]
    url: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)

    # 원본 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizationDecision:
    """벡터화 결정"""
    should_vectorize: bool
    reason: str
    granularity: str = "chunks"  # full, chunks, summary, metadata, none
    priority: int = 5  # 1-10


@dataclass
class StageError:
    """스테이지 오류"""
    stage_name: str
    error_type: str
    message: str
    is_fatal: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """
    파이프라인 컨텍스트

    스테이지 간 데이터 전달 및 상태 관리
    """

    # 원본 아이템
    source_item: SourceItem

    # 처리 상태
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_stage: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # 파싱 결과
    raw_content: Optional[str] = None
    parsed_content: Optional[ParsedContent] = None
    document_type: Optional[DocumentType] = None

    # 추출 결과
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    # 추론 결과
    inferred_metadata: Dict[str, Any] = field(default_factory=dict)

    # 스코어링 결과
    decision_mappings: List[DecisionMapping] = field(default_factory=list)
    overall_relevance: float = 0.0

    # 벡터화 결정
    vectorization_decision: Optional[VectorizationDecision] = None
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # 최종 문서
    document: Optional[Document] = None

    # 제어 플래그
    should_skip: bool = False
    skip_reason: Optional[str] = None

    # 오류
    errors: List[StageError] = field(default_factory=list)

    # 스테이지별 결과 저장
    stage_results: Dict[str, Any] = field(default_factory=dict)

    def mark_stage_start(self, stage_name: str) -> None:
        """스테이지 시작 마킹"""
        self.current_stage = stage_name
        self.status = ProcessingStatus.IN_PROGRESS

    def mark_stage_complete(self, stage_name: str, result: Any = None) -> None:
        """스테이지 완료 마킹"""
        if result is not None:
            self.stage_results[stage_name] = result

    def mark_skip(self, reason: str) -> None:
        """스킵 마킹"""
        self.should_skip = True
        self.skip_reason = reason
        self.status = ProcessingStatus.SKIPPED

    def mark_completed(self) -> None:
        """처리 완료 마킹"""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now()

    def mark_failed(self, error: StageError) -> None:
        """처리 실패 마킹"""
        self.status = ProcessingStatus.FAILED
        self.errors.append(error)
        self.completed_at = datetime.now()

    def add_error(self, error: StageError) -> None:
        """오류 추가"""
        self.errors.append(error)
        if error.is_fatal:
            self.mark_failed(error)

    def has_errors(self) -> bool:
        """오류 존재 여부"""
        return len(self.errors) > 0

    def has_fatal_errors(self) -> bool:
        """치명적 오류 존재 여부"""
        return any(e.is_fatal for e in self.errors)

    @property
    def duration_ms(self) -> int:
        """처리 시간 (밀리초)"""
        end = self.completed_at or datetime.now()
        return int((end - self.started_at).total_seconds() * 1000)

    def to_document(self) -> Document:
        """Document로 변환"""
        if self.document:
            return self.document

        doc = Document(
            source_type=self.source_item.source_type,
            source_id=self.source_item.id,
            document_type=self.document_type or DocumentType.UNKNOWN,
            raw_content=self.raw_content or "",
            parsed_content=self.parsed_content,
        )

        # 메타데이터 설정
        if self.inferred_metadata:
            for key, value in self.inferred_metadata.items():
                if hasattr(doc.metadata, key):
                    setattr(doc.metadata, key, value)

        doc.metadata.url = self.source_item.url
        doc.metadata.raw_metadata = self.source_item.metadata

        # 엔티티/관계 ID
        doc.entity_ids = [e.id for e in self.entities]
        doc.relationship_ids = [r.id for r in self.relationships]

        # 의사결정 매핑
        doc.decision_mapping_ids = [m.id for m in self.decision_mappings]
        doc.overall_relevance_score = self.overall_relevance

        # 청크 ID
        doc.chunk_ids = [c.id for c in self.chunks]

        # 벡터화 상태
        if self.vectorization_decision and self.vectorization_decision.should_vectorize:
            doc.is_vectorized = len(self.chunks) > 0

        self.document = doc
        return doc


@dataclass
class PipelineConfig:
    """파이프라인 설정"""

    # 스테이지 활성화
    enable_parsing: bool = True
    enable_extraction: bool = True
    enable_inference: bool = True
    enable_scoring: bool = True
    enable_vectorization: bool = True

    # 병렬 처리
    max_concurrent: int = 5
    batch_size: int = 10

    # 오류 처리
    continue_on_error: bool = True
    max_retries: int = 3

    # 필터링
    min_content_length: int = 50
    skip_empty: bool = True

    # 벡터화 설정
    vectorization_threshold: float = 0.3  # 최소 관련성 점수

    # 타임아웃
    stage_timeout_seconds: float = 60.0
