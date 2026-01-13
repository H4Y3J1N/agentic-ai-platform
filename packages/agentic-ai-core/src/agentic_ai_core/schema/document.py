"""
Document Schema

문서, 청크, 메타데이터 관련 스키마 정의
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from .base import IdentifiableTimestamped, Scorable, SchemaBase


class SourceType(str, Enum):
    """데이터 소스 타입"""
    NOTION = "notion"
    SLACK = "slack"
    CONFLUENCE = "confluence"
    GOOGLE_DOCS = "google_docs"
    GITHUB = "github"
    LOCAL_FILE = "local_file"
    WEB = "web"
    API = "api"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """문서 타입 (파서 매핑용)"""
    POLICY = "policy"                  # 정책/규정 문서
    MEETING_NOTE = "meeting_note"      # 회의록
    FAQ = "faq"                        # FAQ
    TECHNICAL_DOC = "technical_doc"    # 기술 문서
    ANNOUNCEMENT = "announcement"      # 공지사항
    CONVERSATION = "conversation"      # 대화 (Slack 스레드)
    WIKI = "wiki"                      # 위키 페이지
    CODE = "code"                      # 코드/스크립트
    REPORT = "report"                  # 보고서
    TEMPLATE = "template"              # 템플릿
    UNKNOWN = "unknown"


class ResolutionLevel(str, Enum):
    """데이터 해상도 레벨 (생명주기 관리용)"""
    FULL = "full"                      # 전체 데이터 유지
    SUMMARY = "summary"                # 요약만 유지
    METADATA_ONLY = "metadata_only"    # 메타데이터만 유지
    ARCHIVED = "archived"              # 아카이브 (저비용 스토리지)
    DELETED = "deleted"                # 삭제됨


class DocumentMetadata(SchemaBase):
    """문서 메타데이터 (자동 추론 + 수동 입력)"""

    # 기본 정보
    title: str = ""
    author: Optional[str] = None
    department: Optional[str] = None
    url: Optional[str] = None

    # 자동 추출
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    language: str = "ko"

    # 분석 결과
    sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # 동적 메트릭
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    access_frequency: int = Field(default=0, ge=0)
    citation_count: int = Field(default=0, ge=0)

    # 원본 메타데이터 (소스별 다름)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedSection(SchemaBase):
    """파싱된 문서 섹션"""

    name: str
    content: str
    level: int = 0  # 헤더 레벨 (0=최상위)
    children: List["ParsedSection"] = Field(default_factory=list)


class ParsedContent(SchemaBase):
    """파싱된 문서 콘텐츠"""

    # 섹션 구조
    sections: List[ParsedSection] = Field(default_factory=list)

    # 구조화된 데이터 (회의록의 액션 아이템 등)
    structured_data: Dict[str, Any] = Field(default_factory=dict)

    # 코드 블록 (기술 문서용)
    code_blocks: List[Dict[str, str]] = Field(default_factory=list)

    # 테이블 데이터
    tables: List[List[List[str]]] = Field(default_factory=list)

    def get_plain_text(self) -> str:
        """전체 텍스트 추출"""
        texts = []
        for section in self.sections:
            texts.append(self._extract_section_text(section))
        return "\n\n".join(texts)

    def _extract_section_text(self, section: ParsedSection) -> str:
        """섹션에서 텍스트 추출 (재귀)"""
        result = section.content
        for child in section.children:
            result += "\n" + self._extract_section_text(child)
        return result


class Chunk(IdentifiableTimestamped, Scorable):
    """문서 청크 (벡터화 단위)"""

    # 소속 문서
    document_id: str

    # 콘텐츠
    content: str
    token_count: int = 0

    # 위치 정보
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0

    # 컨텍스트 (헤더 경로 등)
    header_context: str = ""
    section_name: Optional[str] = None

    # 벡터
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    # 메타데이터 (문서에서 상속)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def has_embedding(self) -> bool:
        """임베딩 존재 여부"""
        return self.embedding is not None and len(self.embedding) > 0


class Document(IdentifiableTimestamped):
    """문서 스키마"""

    # 소스 정보
    source_type: SourceType = SourceType.UNKNOWN
    source_id: str = ""  # 원본 시스템의 ID (예: Notion page ID)

    # 문서 타입
    document_type: DocumentType = DocumentType.UNKNOWN

    # 콘텐츠
    raw_content: str = ""
    parsed_content: Optional[ParsedContent] = None

    # 메타데이터
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    # 의사결정 관련성 (decision.py에서 정의)
    decision_mapping_ids: List[str] = Field(default_factory=list)
    overall_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # 생명주기
    resolution_level: ResolutionLevel = ResolutionLevel.FULL
    last_accessed_at: Optional[datetime] = None

    # 그래프 연결 (entity.py에서 정의)
    entity_ids: List[str] = Field(default_factory=list)
    relationship_ids: List[str] = Field(default_factory=list)

    # 청크 (별도 저장, 참조만)
    chunk_ids: List[str] = Field(default_factory=list)

    # 벡터화 상태
    is_vectorized: bool = False
    vectorized_at: Optional[datetime] = None

    @property
    def title(self) -> str:
        """문서 제목"""
        return self.metadata.title or f"Untitled ({self.source_type.value})"

    @property
    def url(self) -> Optional[str]:
        """문서 URL"""
        return self.metadata.url

    def mark_accessed(self) -> None:
        """접근 기록"""
        self.last_accessed_at = datetime.now()
        self.metadata.access_frequency += 1

    def mark_vectorized(self, chunk_ids: List[str]) -> None:
        """벡터화 완료 마킹"""
        self.is_vectorized = True
        self.vectorized_at = datetime.now()
        self.chunk_ids = chunk_ids

    @classmethod
    def from_notion(cls, page_data: Dict[str, Any]) -> "Document":
        """Notion 페이지에서 생성"""
        return cls(
            source_type=SourceType.NOTION,
            source_id=page_data.get("id", ""),
            metadata=DocumentMetadata(
                title=page_data.get("title", ""),
                url=page_data.get("url", ""),
                raw_metadata=page_data.get("properties", {})
            )
        )

    @classmethod
    def from_slack(cls, message_data: Dict[str, Any]) -> "Document":
        """Slack 메시지에서 생성"""
        return cls(
            source_type=SourceType.SLACK,
            source_id=message_data.get("ts", ""),
            document_type=DocumentType.CONVERSATION,
            metadata=DocumentMetadata(
                title=f"Slack: {message_data.get('channel', 'unknown')}",
                raw_metadata=message_data
            )
        )
