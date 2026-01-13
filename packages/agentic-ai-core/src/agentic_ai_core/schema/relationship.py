"""
Relationship Schema

엔티티 간 관계 스키마 정의
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import Field

from .base import IdentifiableTimestamped, Scorable, Provenanced, SchemaBase


class RelationType(str, Enum):
    """관계 타입"""

    # 조직 관계
    BELONGS_TO = "belongs_to"          # 소속 (person → team)
    MANAGES = "manages"                # 관리 (person → team/project)
    REPORTS_TO = "reports_to"          # 보고 (person → person)
    MEMBER_OF = "member_of"            # 멤버 (person → team)

    # 프로젝트/업무 관계
    WORKS_ON = "works_on"              # 참여 (person → project)
    OWNS = "owns"                      # 소유 (person/team → project/product)
    CONTRIBUTES_TO = "contributes_to"  # 기여 (person → project/document)
    RESPONSIBLE_FOR = "responsible_for"  # 담당 (person/team → task)

    # 문서 관계
    REFERENCES = "references"          # 참조 (document → document)
    SUPERSEDES = "supersedes"          # 대체 (document → document, 신규 → 구버전)
    RELATED_TO = "related_to"          # 관련 (general)
    DERIVED_FROM = "derived_from"      # 파생 (document → document)
    MENTIONS = "mentions"              # 언급 (document → entity)

    # 프로세스/의존성 관계
    DEPENDS_ON = "depends_on"          # 의존 (project → project)
    TRIGGERS = "triggers"              # 트리거 (event → process)
    PART_OF = "part_of"                # 구성요소 (task → project)
    PRECEDES = "precedes"              # 선행 (task → task)
    FOLLOWS = "follows"                # 후행 (task → task)

    # 정책/규칙 관계
    APPLIES_TO = "applies_to"          # 적용 (policy → department/role)
    GOVERNED_BY = "governed_by"        # 규제 (process → policy)
    COMPLIES_WITH = "complies_with"    # 준수 (project → policy)

    # 기술 관계
    USES = "uses"                      # 사용 (project → tool)
    INTEGRATES_WITH = "integrates_with"  # 연동 (system → system)
    HOSTS = "hosts"                    # 호스팅 (system → service)

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
    weight: float = Field(default=1.0, ge=0.0)  # 관계 강도

    # 방향성
    is_bidirectional: bool = False  # True면 양방향 관계

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

    def set_validity(
        self,
        from_date: Optional[datetime] = None,
        until_date: Optional[datetime] = None
    ) -> None:
        """유효 기간 설정"""
        self.valid_from = from_date
        self.valid_until = until_date

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
    def create_belongs_to(
        cls,
        member_id: str,
        group_id: str,
        **kwargs
    ) -> "Relationship":
        """소속 관계 생성 헬퍼"""
        return cls(
            relation_type=RelationType.BELONGS_TO,
            source_entity_id=member_id,
            target_entity_id=group_id,
            **kwargs
        )

    @classmethod
    def create_manages(
        cls,
        manager_id: str,
        managed_id: str,
        **kwargs
    ) -> "Relationship":
        """관리 관계 생성 헬퍼"""
        return cls(
            relation_type=RelationType.MANAGES,
            source_entity_id=manager_id,
            target_entity_id=managed_id,
            **kwargs
        )

    @classmethod
    def create_references(
        cls,
        source_doc_id: str,
        target_doc_id: str,
        **kwargs
    ) -> "Relationship":
        """문서 참조 관계 생성 헬퍼"""
        return cls(
            relation_type=RelationType.REFERENCES,
            source_entity_id=source_doc_id,
            target_entity_id=target_doc_id,
            **kwargs
        )

    @classmethod
    def create_works_on(
        cls,
        person_id: str,
        project_id: str,
        role: Optional[str] = None,
        **kwargs
    ) -> "Relationship":
        """프로젝트 참여 관계 생성 헬퍼"""
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
    # ... 추가 메타데이터
}
