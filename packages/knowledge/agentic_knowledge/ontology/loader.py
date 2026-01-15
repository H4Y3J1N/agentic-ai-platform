"""
Ontology Loader

YAML 기반 온톨로지 로더
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OntologyFormat(Enum):
    """온톨로지 포맷"""
    YAML = "yaml"
    JSON = "json"


@dataclass
class EntityTypeDefinition:
    """엔티티 타입 정의"""
    name: str
    description: str = ""
    parent: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    required_properties: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class RelationTypeDefinition:
    """관계 타입 정의"""
    name: str
    description: str = ""
    source_types: List[str] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    inverse_name: Optional[str] = None


@dataclass
class DecisionTypeDefinition:
    """의사결정 타입 정의"""
    name: str
    description: str = ""
    category: str = ""
    frequency: str = "weekly"
    impact_level: str = "medium"
    relevant_entity_types: List[str] = field(default_factory=list)
    relevant_document_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class Ontology:
    """온톨로지 전체 구조"""
    name: str
    version: str = "1.0.0"
    description: str = ""

    # 타입 정의
    entity_types: Dict[str, EntityTypeDefinition] = field(default_factory=dict)
    relation_types: Dict[str, RelationTypeDefinition] = field(default_factory=dict)
    decision_types: Dict[str, DecisionTypeDefinition] = field(default_factory=dict)

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """엔티티 타입 조회"""
        return self.entity_types.get(name)

    def get_relation_type(self, name: str) -> Optional[RelationTypeDefinition]:
        """관계 타입 조회"""
        return self.relation_types.get(name)

    def get_decision_type(self, name: str) -> Optional[DecisionTypeDefinition]:
        """의사결정 타입 조회"""
        return self.decision_types.get(name)

    def list_entity_types(self) -> List[str]:
        """엔티티 타입 목록"""
        return list(self.entity_types.keys())

    def list_relation_types(self) -> List[str]:
        """관계 타입 목록"""
        return list(self.relation_types.keys())

    def list_decision_types(self) -> List[str]:
        """의사결정 타입 목록"""
        return list(self.decision_types.keys())

    def get_entity_hierarchy(self) -> Dict[str, List[str]]:
        """엔티티 타입 계층 구조"""
        hierarchy: Dict[str, List[str]] = {}

        for name, entity_type in self.entity_types.items():
            parent = entity_type.parent or "root"
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(name)

        return hierarchy

    def get_valid_relations_for(
        self,
        source_type: str
    ) -> List[RelationTypeDefinition]:
        """특정 소스 타입에 유효한 관계들"""
        valid = []
        for rel in self.relation_types.values():
            if not rel.source_types or source_type in rel.source_types:
                valid.append(rel)
        return valid


class OntologyLoader:
    """온톨로지 로더"""

    def __init__(self):
        self._cache: Dict[str, Ontology] = {}

    def load(
        self,
        source: Union[str, Path, Dict[str, Any]],
        format: Optional[OntologyFormat] = None
    ) -> Ontology:
        """
        온톨로지 로드

        Args:
            source: 파일 경로, URL, 또는 딕셔너리
            format: 포맷 (자동 감지)

        Returns:
            로드된 온톨로지
        """
        # 딕셔너리인 경우 직접 파싱
        if isinstance(source, dict):
            return self._parse_ontology(source)

        # 파일 경로인 경우
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {path}")

        # 캐시 확인
        cache_key = str(path.absolute())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 포맷 감지
        if format is None:
            if path.suffix in ['.yaml', '.yml']:
                format = OntologyFormat.YAML
            elif path.suffix == '.json':
                format = OntologyFormat.JSON
            else:
                format = OntologyFormat.YAML

        # 파일 로드
        data = self._load_file(path, format)
        ontology = self._parse_ontology(data)

        # 캐시 저장
        self._cache[cache_key] = ontology

        return ontology

    def load_from_string(
        self,
        content: str,
        format: OntologyFormat = OntologyFormat.YAML
    ) -> Ontology:
        """문자열에서 온톨로지 로드"""
        data = self._parse_string(content, format)
        return self._parse_ontology(data)

    def _load_file(
        self,
        path: Path,
        format: OntologyFormat
    ) -> Dict[str, Any]:
        """파일 로드"""
        content = path.read_text(encoding='utf-8')
        return self._parse_string(content, format)

    def _parse_string(
        self,
        content: str,
        format: OntologyFormat
    ) -> Dict[str, Any]:
        """문자열 파싱"""
        if format == OntologyFormat.YAML:
            try:
                import yaml
                return yaml.safe_load(content) or {}
            except ImportError:
                raise ImportError("PyYAML required. Install with: pip install pyyaml")

        elif format == OntologyFormat.JSON:
            import json
            return json.loads(content)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _parse_ontology(self, data: Dict[str, Any]) -> Ontology:
        """온톨로지 파싱"""
        # 기본 정보
        name = data.get("name", "default")
        version = data.get("version", "1.0.0")
        description = data.get("description", "")

        # 엔티티 타입 파싱
        entity_types = {}
        for et_name, et_data in data.get("entity_types", {}).items():
            entity_types[et_name] = self._parse_entity_type(et_name, et_data)

        # 관계 타입 파싱
        relation_types = {}
        for rt_name, rt_data in data.get("relation_types", {}).items():
            relation_types[rt_name] = self._parse_relation_type(rt_name, rt_data)

        # 의사결정 타입 파싱
        decision_types = {}
        for dt_name, dt_data in data.get("decision_types", {}).items():
            decision_types[dt_name] = self._parse_decision_type(dt_name, dt_data)

        return Ontology(
            name=name,
            version=version,
            description=description,
            entity_types=entity_types,
            relation_types=relation_types,
            decision_types=decision_types,
            metadata=data.get("metadata", {})
        )

    def _parse_entity_type(
        self,
        name: str,
        data: Union[Dict[str, Any], str]
    ) -> EntityTypeDefinition:
        """엔티티 타입 파싱"""
        if isinstance(data, str):
            return EntityTypeDefinition(name=name, description=data)

        return EntityTypeDefinition(
            name=name,
            description=data.get("description", ""),
            parent=data.get("parent"),
            properties=data.get("properties", {}),
            required_properties=data.get("required_properties", []),
            keywords=data.get("keywords", []),
            examples=data.get("examples", [])
        )

    def _parse_relation_type(
        self,
        name: str,
        data: Union[Dict[str, Any], str]
    ) -> RelationTypeDefinition:
        """관계 타입 파싱"""
        if isinstance(data, str):
            return RelationTypeDefinition(name=name, description=data)

        return RelationTypeDefinition(
            name=name,
            description=data.get("description", ""),
            source_types=data.get("source_types", []),
            target_types=data.get("target_types", []),
            properties=data.get("properties", {}),
            bidirectional=data.get("bidirectional", False),
            inverse_name=data.get("inverse_name")
        )

    def _parse_decision_type(
        self,
        name: str,
        data: Union[Dict[str, Any], str]
    ) -> DecisionTypeDefinition:
        """의사결정 타입 파싱"""
        if isinstance(data, str):
            return DecisionTypeDefinition(name=name, description=data)

        return DecisionTypeDefinition(
            name=name,
            description=data.get("description", ""),
            category=data.get("category", ""),
            frequency=data.get("frequency", "weekly"),
            impact_level=data.get("impact_level", "medium"),
            relevant_entity_types=data.get("relevant_entity_types", []),
            relevant_document_types=data.get("relevant_document_types", []),
            keywords=data.get("keywords", [])
        )

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()


class OntologyMerger:
    """온톨로지 병합기"""

    def merge(
        self,
        base: Ontology,
        *extensions: Ontology
    ) -> Ontology:
        """
        여러 온톨로지 병합

        Args:
            base: 기본 온톨로지
            extensions: 확장 온톨로지들

        Returns:
            병합된 온톨로지
        """
        merged_entities = dict(base.entity_types)
        merged_relations = dict(base.relation_types)
        merged_decisions = dict(base.decision_types)
        merged_metadata = dict(base.metadata)

        for ext in extensions:
            # 엔티티 병합 (덮어쓰기)
            merged_entities.update(ext.entity_types)

            # 관계 병합
            merged_relations.update(ext.relation_types)

            # 의사결정 병합
            merged_decisions.update(ext.decision_types)

            # 메타데이터 병합
            merged_metadata.update(ext.metadata)

        return Ontology(
            name=f"{base.name}_merged",
            version=base.version,
            description=f"Merged ontology: {base.name}",
            entity_types=merged_entities,
            relation_types=merged_relations,
            decision_types=merged_decisions,
            metadata=merged_metadata
        )


# 기본 온톨로지 정의 (코드 내장)
DEFAULT_ONTOLOGY_YAML = """
name: internal_ops
version: "1.0.0"
description: "Internal Operations 기본 온톨로지"

entity_types:
  person:
    description: "조직 구성원"
    properties:
      name: string
      email: string
      department: string
      role: string
    required_properties: [name]
    keywords: [사람, 직원, 담당자, person, employee, staff]

  team:
    description: "팀/부서"
    properties:
      name: string
      manager: person
    keywords: [팀, 부서, 조직, team, department]

  project:
    description: "프로젝트"
    properties:
      name: string
      status: string
      deadline: date
    keywords: [프로젝트, 과제, project]

  policy:
    description: "정책/규정"
    properties:
      name: string
      effective_date: date
      category: string
    keywords: [정책, 규정, 규칙, policy, rule]

  tool:
    description: "도구/시스템"
    properties:
      name: string
      url: string
      category: string
    keywords: [도구, 시스템, 툴, tool, system]

  process:
    description: "업무 프로세스"
    properties:
      name: string
      steps: list
    keywords: [프로세스, 절차, 방법, process, procedure]

relation_types:
  belongs_to:
    description: "소속 관계"
    source_types: [person]
    target_types: [team, project]

  manages:
    description: "관리 관계"
    source_types: [person]
    target_types: [team, project]
    inverse_name: managed_by

  works_on:
    description: "작업 관계"
    source_types: [person]
    target_types: [project]

  uses:
    description: "사용 관계"
    source_types: [person, team]
    target_types: [tool]

  governs:
    description: "적용 관계"
    source_types: [policy]
    target_types: [process, team]

  related_to:
    description: "일반 관련 관계"
    bidirectional: true

decision_types:
  leave_request:
    description: "휴가 신청 관련 의사결정"
    category: hr
    frequency: daily
    impact_level: low
    relevant_entity_types: [person, policy]
    relevant_document_types: [policy, faq]
    keywords: [휴가, 연차, 반차, leave, vacation]

  onboarding:
    description: "신규 입사자 온보딩"
    category: hr
    frequency: weekly
    impact_level: medium
    relevant_entity_types: [person, team, process, tool]
    relevant_document_types: [policy, wiki, technical_doc]
    keywords: [온보딩, 입사, 신규, onboarding]

  expense_approval:
    description: "비용 승인"
    category: finance
    frequency: weekly
    impact_level: medium
    relevant_entity_types: [person, project, policy]
    relevant_document_types: [policy, faq]
    keywords: [비용, 경비, 승인, expense, budget]

  project_planning:
    description: "프로젝트 계획"
    category: project
    frequency: monthly
    impact_level: high
    relevant_entity_types: [project, person, team]
    relevant_document_types: [technical_doc, meeting_note]
    keywords: [프로젝트, 계획, 일정, project, planning]
"""


def load_default_ontology() -> Ontology:
    """기본 온톨로지 로드"""
    loader = OntologyLoader()
    return loader.load_from_string(DEFAULT_ONTOLOGY_YAML, OntologyFormat.YAML)
