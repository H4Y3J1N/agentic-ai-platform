"""
Entity Schema

지식 그래프 엔티티 관련 스키마 정의
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from .base import IdentifiableTimestamped, Scorable, Provenanced, SchemaBase


class EntityType(str, Enum):
    """엔티티 타입"""

    # 조직 관련
    PERSON = "person"
    TEAM = "team"
    DEPARTMENT = "department"
    ORGANIZATION = "organization"

    # 업무 관련
    PROJECT = "project"
    PRODUCT = "product"
    SERVICE = "service"
    TASK = "task"

    # 문서/프로세스
    PROCESS = "process"
    POLICY = "policy"
    DOCUMENT = "document"

    # 기술 관련
    TOOL = "tool"
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"

    # 추상 개념
    CONCEPT = "concept"
    TOPIC = "topic"
    SKILL = "skill"

    # 시간/장소
    EVENT = "event"
    LOCATION = "location"
    DATE = "date"

    # 기타
    UNKNOWN = "unknown"


class EntityRef(SchemaBase):
    """엔티티 참조 (경량 버전)"""

    id: str
    entity_type: EntityType
    name: str

    def __hash__(self) -> int:
        return hash(self.id)


class Entity(IdentifiableTimestamped, Scorable, Provenanced):
    """지식 그래프 엔티티"""

    # 기본 정보
    entity_type: EntityType = EntityType.UNKNOWN
    name: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    # 속성 (타입별 다름)
    properties: Dict[str, Any] = Field(default_factory=dict)

    # 벡터화 (선택적)
    embedding: Optional[List[float]] = None
    should_vectorize: bool = False

    # 의사결정 영향도
    decision_influence: Dict[str, float] = Field(default_factory=dict)

    # 정규화된 이름 (검색/매칭용)
    normalized_name: str = ""

    # 관계 (relationship.py와 연결)
    outgoing_relationship_ids: List[str] = Field(default_factory=list)
    incoming_relationship_ids: List[str] = Field(default_factory=list)

    # 멘션된 문서들
    mentioned_in_document_ids: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """초기화 후 처리"""
        if not self.normalized_name:
            self.normalized_name = self._normalize(self.name)

    def _normalize(self, text: str) -> str:
        """이름 정규화"""
        return text.lower().strip()

    def matches(self, query: str) -> bool:
        """쿼리와 매칭되는지 확인"""
        normalized_query = self._normalize(query)
        if normalized_query == self.normalized_name:
            return True
        return any(
            self._normalize(alias) == normalized_query
            for alias in self.aliases
        )

    def add_alias(self, alias: str) -> None:
        """별칭 추가"""
        normalized = self._normalize(alias)
        if normalized != self.normalized_name and alias not in self.aliases:
            self.aliases.append(alias)

    def add_mention(self, document_id: str) -> None:
        """문서 멘션 추가"""
        if document_id not in self.mentioned_in_document_ids:
            self.mentioned_in_document_ids.append(document_id)

    def set_decision_influence(self, decision_id: str, score: float) -> None:
        """의사결정 영향도 설정"""
        self.decision_influence[decision_id] = max(0.0, min(1.0, score))

    def get_decision_influence(self, decision_id: str) -> float:
        """의사결정 영향도 조회"""
        return self.decision_influence.get(decision_id, 0.0)

    def to_ref(self) -> EntityRef:
        """경량 참조로 변환"""
        return EntityRef(
            id=self.id,
            entity_type=self.entity_type,
            name=self.name
        )

    @classmethod
    def create_person(
        cls,
        name: str,
        email: Optional[str] = None,
        department: Optional[str] = None,
        **kwargs
    ) -> "Entity":
        """Person 엔티티 생성 헬퍼"""
        properties = {"email": email, "department": department}
        properties = {k: v for k, v in properties.items() if v is not None}
        return cls(
            entity_type=EntityType.PERSON,
            name=name,
            properties=properties,
            **kwargs
        )

    @classmethod
    def create_project(
        cls,
        name: str,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        **kwargs
    ) -> "Entity":
        """Project 엔티티 생성 헬퍼"""
        properties = {"status": status, "owner": owner}
        properties = {k: v for k, v in properties.items() if v is not None}
        return cls(
            entity_type=EntityType.PROJECT,
            name=name,
            properties=properties,
            **kwargs
        )

    @classmethod
    def create_policy(
        cls,
        name: str,
        category: Optional[str] = None,
        effective_date: Optional[datetime] = None,
        **kwargs
    ) -> "Entity":
        """Policy 엔티티 생성 헬퍼"""
        properties = {
            "category": category,
            "effective_date": effective_date.isoformat() if effective_date else None
        }
        properties = {k: v for k, v in properties.items() if v is not None}
        return cls(
            entity_type=EntityType.POLICY,
            name=name,
            properties=properties,
            should_vectorize=True,  # 정책은 기본적으로 벡터화
            **kwargs
        )
