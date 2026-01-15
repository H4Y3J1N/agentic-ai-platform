"""
Relationship Schema

엔티티 간 관계 스키마 정의

사용 시점:
- 엔티티 간 관계를 정의할 때 (belongs_to, manages, works_on 등)
- 지식 그래프 엣지로 사용할 때
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from agentic_ai_core.schema import (
    IdentifiableTimestamped,
    Scorable,
    Provenanced,
    SchemaBase,
)


class RelationType(str, Enum):
    """관계 타입"""

    # 조직 관계
    BELONGS_TO = "belongs_to"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    MEMBER_OF = "member_of"

    # 프로젝트/업무 관계
    WORKS_ON = "works_on"
    OWNS = "owns"
    CONTRIBUTES_TO = "contributes_to"
    RESPONSIBLE_FOR = "responsible_for"

    # 문서 관계
    REFERENCES = "references"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    MENTIONS = "mentions"

    # 프로세스/의존성 관계
    DEPENDS_ON = "depends_on"
    TRIGGERS = "triggers"
    PART_OF = "part_of"
    PRECEDES = "precedes"
    FOLLOWS = "follows"

    # 정책/규칙 관계
    APPLIES_TO = "applies_to"
    GOVERNED_BY = "governed_by"
    COMPLIES_WITH = "complies_with"

    # 기술 관계
    USES = "uses"
    INTEGRATES_WITH = "integrates_with"
    HOSTS = "hosts"

    # 기타
    UNKNOWN = "unknown"


class RelationshipRef(SchemaBase):
    """관계 참조 (경량 버전)"""

    id: str
    relation_type: RelationType
    source_entity_id: str
    target_entity_id: str

    def __hash__(self) -> int:
        return hash(self.id)


class Relationship(IdentifiableTimestamped, Scorable, Provenanced):
    """엔티티 간 관계"""

    # 관계 정의
    relation_type: RelationType = RelationType.UNKNOWN
    source_entity_id: str
    target_entity_id: str

    # 관계 속성
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0)

    # 방향성
    is_bidirectional: bool = False

    # 시간적 유효성
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # 설명
    description: Optional[str] = None

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """특정 시점에 유효한 관계인지 확인"""
        check_time = at_time or datetime.now()

        if self.valid_from and check_time < self.valid_from:
            return False
        if self.valid_until and check_time > self.valid_until:
            return False

        return True

    def is_expired(self) -> bool:
        """만료된 관계인지 확인"""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until

    def to_ref(self) -> RelationshipRef:
        """경량 참조로 변환"""
        return RelationshipRef(
            id=self.id,
            relation_type=self.relation_type,
            source_entity_id=self.source_entity_id,
            target_entity_id=self.target_entity_id
        )

    def involves(self, entity_id: str) -> bool:
        """특정 엔티티가 관계에 포함되는지"""
        return entity_id in (self.source_entity_id, self.target_entity_id)

    def get_other_entity(self, entity_id: str) -> Optional[str]:
        """주어진 엔티티의 상대방 엔티티 ID 반환"""
        if entity_id == self.source_entity_id:
            return self.target_entity_id
        elif entity_id == self.target_entity_id:
            return self.source_entity_id
        return None

    @classmethod
    def create_belongs_to(cls, member_id: str, group_id: str, **kwargs) -> "Relationship":
        """소속 관계 생성"""
        return cls(
            relation_type=RelationType.BELONGS_TO,
            source_entity_id=member_id,
            target_entity_id=group_id,
            **kwargs
        )

    @classmethod
    def create_manages(cls, manager_id: str, managed_id: str, **kwargs) -> "Relationship":
        """관리 관계 생성"""
        return cls(
            relation_type=RelationType.MANAGES,
            source_entity_id=manager_id,
            target_entity_id=managed_id,
            **kwargs
        )

    @classmethod
    def create_works_on(cls, person_id: str, project_id: str, role: Optional[str] = None, **kwargs) -> "Relationship":
        """프로젝트 참여 관계 생성"""
        properties = {"role": role} if role else {}
        return cls(
            relation_type=RelationType.WORKS_ON,
            source_entity_id=person_id,
            target_entity_id=project_id,
            properties=properties,
            **kwargs
        )


# 관계 타입별 메타데이터
RELATION_TYPE_METADATA: Dict[RelationType, Dict[str, Any]] = {
    RelationType.BELONGS_TO: {
        "source_types": ["person", "team", "project"],
        "target_types": ["team", "department", "organization"],
        "is_hierarchical": True,
    },
    RelationType.MANAGES: {
        "source_types": ["person"],
        "target_types": ["team", "project", "person"],
        "is_hierarchical": True,
    },
    RelationType.WORKS_ON: {
        "source_types": ["person"],
        "target_types": ["project", "task"],
        "is_hierarchical": False,
    },
    RelationType.REFERENCES: {
        "source_types": ["document"],
        "target_types": ["document", "policy"],
        "is_hierarchical": False,
    },
}
