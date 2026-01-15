"""
Ontology Validator

온톨로지 스키마 검증
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from .loader import (
    Ontology,
    EntityTypeDefinition,
    RelationTypeDefinition,
    DecisionTypeDefinition,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """검증 심각도"""
    ERROR = "error"       # 치명적 오류
    WARNING = "warning"   # 경고
    INFO = "info"         # 정보


@dataclass
class ValidationIssue:
    """검증 이슈"""
    severity: ValidationSeverity
    code: str
    message: str
    path: str = ""
    suggestion: str = ""


@dataclass
class ValidationResult:
    """검증 결과"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_at: str = ""

    @property
    def errors(self) -> List[ValidationIssue]:
        """에러만 반환"""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """경고만 반환"""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: str = ""
    ) -> None:
        """에러 추가"""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
            path=path,
            suggestion=suggestion
        ))
        self.valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        path: str = "",
        suggestion: str = ""
    ) -> None:
        """경고 추가"""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code=code,
            message=message,
            path=path,
            suggestion=suggestion
        ))

    def add_info(
        self,
        code: str,
        message: str,
        path: str = ""
    ) -> None:
        """정보 추가"""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            code=code,
            message=message,
            path=path
        ))

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                    "suggestion": i.suggestion
                }
                for i in self.issues
            ]
        }


class OntologyValidator:
    """온톨로지 검증기"""

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 엄격 모드 (경고도 에러로 처리)
        """
        self.strict = strict

    def validate(self, ontology: Ontology) -> ValidationResult:
        """
        온톨로지 검증

        Args:
            ontology: 검증할 온톨로지

        Returns:
            검증 결과
        """
        from datetime import datetime

        result = ValidationResult(
            valid=True,
            validated_at=datetime.now().isoformat()
        )

        # 기본 검증
        self._validate_basic(ontology, result)

        # 엔티티 타입 검증
        self._validate_entity_types(ontology, result)

        # 관계 타입 검증
        self._validate_relation_types(ontology, result)

        # 의사결정 타입 검증
        self._validate_decision_types(ontology, result)

        # 참조 무결성 검증
        self._validate_references(ontology, result)

        # 엄격 모드에서 경고 처리
        if self.strict and result.warnings:
            for warning in result.warnings:
                result.add_error(
                    code=warning.code,
                    message=f"[Strict] {warning.message}",
                    path=warning.path,
                    suggestion=warning.suggestion
                )

        return result

    def _validate_basic(
        self,
        ontology: Ontology,
        result: ValidationResult
    ) -> None:
        """기본 검증"""
        # 이름 검증
        if not ontology.name:
            result.add_error(
                code="MISSING_NAME",
                message="온톨로지 이름이 없습니다",
                path="name",
                suggestion="name 필드를 추가하세요"
            )

        # 버전 검증
        if not ontology.version:
            result.add_warning(
                code="MISSING_VERSION",
                message="버전 정보가 없습니다",
                path="version",
                suggestion="version 필드를 추가하세요 (예: 1.0.0)"
            )

        # 최소 타입 검증
        if not ontology.entity_types:
            result.add_warning(
                code="NO_ENTITY_TYPES",
                message="엔티티 타입이 정의되지 않았습니다",
                path="entity_types",
                suggestion="최소 하나의 엔티티 타입을 정의하세요"
            )

    def _validate_entity_types(
        self,
        ontology: Ontology,
        result: ValidationResult
    ) -> None:
        """엔티티 타입 검증"""
        for name, entity_type in ontology.entity_types.items():
            path = f"entity_types.{name}"

            # 이름 규칙 검증
            if not self._is_valid_identifier(name):
                result.add_error(
                    code="INVALID_ENTITY_NAME",
                    message=f"유효하지 않은 엔티티 타입 이름: {name}",
                    path=path,
                    suggestion="영문 소문자, 숫자, 언더스코어만 사용하세요"
                )

            # 부모 타입 존재 검증
            if entity_type.parent:
                if entity_type.parent not in ontology.entity_types:
                    result.add_error(
                        code="INVALID_PARENT",
                        message=f"존재하지 않는 부모 타입: {entity_type.parent}",
                        path=f"{path}.parent",
                        suggestion=f"유효한 엔티티 타입을 지정하세요: {list(ontology.entity_types.keys())}"
                    )

            # 순환 상속 검증
            if entity_type.parent:
                if self._has_circular_inheritance(name, ontology):
                    result.add_error(
                        code="CIRCULAR_INHERITANCE",
                        message=f"순환 상속이 감지되었습니다: {name}",
                        path=f"{path}.parent"
                    )

            # 필수 속성 검증
            for req_prop in entity_type.required_properties:
                if req_prop not in entity_type.properties:
                    result.add_warning(
                        code="MISSING_REQUIRED_PROPERTY",
                        message=f"필수 속성이 properties에 정의되지 않음: {req_prop}",
                        path=f"{path}.required_properties",
                        suggestion=f"properties에 {req_prop}을 정의하세요"
                    )

            # 설명 검증
            if not entity_type.description:
                result.add_info(
                    code="MISSING_DESCRIPTION",
                    message=f"엔티티 타입 설명이 없습니다: {name}",
                    path=path
                )

    def _validate_relation_types(
        self,
        ontology: Ontology,
        result: ValidationResult
    ) -> None:
        """관계 타입 검증"""
        for name, rel_type in ontology.relation_types.items():
            path = f"relation_types.{name}"

            # 이름 규칙 검증
            if not self._is_valid_identifier(name):
                result.add_error(
                    code="INVALID_RELATION_NAME",
                    message=f"유효하지 않은 관계 타입 이름: {name}",
                    path=path
                )

            # 소스/타겟 타입 검증
            for source in rel_type.source_types:
                if source not in ontology.entity_types:
                    result.add_error(
                        code="INVALID_SOURCE_TYPE",
                        message=f"존재하지 않는 소스 엔티티 타입: {source}",
                        path=f"{path}.source_types"
                    )

            for target in rel_type.target_types:
                if target not in ontology.entity_types:
                    result.add_error(
                        code="INVALID_TARGET_TYPE",
                        message=f"존재하지 않는 타겟 엔티티 타입: {target}",
                        path=f"{path}.target_types"
                    )

            # 역관계 검증
            if rel_type.inverse_name:
                if rel_type.inverse_name not in ontology.relation_types:
                    result.add_warning(
                        code="MISSING_INVERSE",
                        message=f"역관계가 정의되지 않음: {rel_type.inverse_name}",
                        path=f"{path}.inverse_name",
                        suggestion=f"relation_types에 {rel_type.inverse_name}을 정의하세요"
                    )

    def _validate_decision_types(
        self,
        ontology: Ontology,
        result: ValidationResult
    ) -> None:
        """의사결정 타입 검증"""
        valid_frequencies = {"daily", "weekly", "monthly", "quarterly", "yearly", "rare"}
        valid_impacts = {"low", "medium", "high", "critical"}

        for name, dec_type in ontology.decision_types.items():
            path = f"decision_types.{name}"

            # 이름 규칙 검증
            if not self._is_valid_identifier(name):
                result.add_error(
                    code="INVALID_DECISION_NAME",
                    message=f"유효하지 않은 의사결정 타입 이름: {name}",
                    path=path
                )

            # 빈도 검증
            if dec_type.frequency not in valid_frequencies:
                result.add_warning(
                    code="INVALID_FREQUENCY",
                    message=f"유효하지 않은 빈도: {dec_type.frequency}",
                    path=f"{path}.frequency",
                    suggestion=f"유효한 값: {valid_frequencies}"
                )

            # 영향 수준 검증
            if dec_type.impact_level not in valid_impacts:
                result.add_warning(
                    code="INVALID_IMPACT",
                    message=f"유효하지 않은 영향 수준: {dec_type.impact_level}",
                    path=f"{path}.impact_level",
                    suggestion=f"유효한 값: {valid_impacts}"
                )

            # 관련 엔티티 타입 검증
            for entity_type in dec_type.relevant_entity_types:
                if entity_type not in ontology.entity_types:
                    result.add_warning(
                        code="UNKNOWN_ENTITY_TYPE",
                        message=f"알 수 없는 엔티티 타입 참조: {entity_type}",
                        path=f"{path}.relevant_entity_types"
                    )

            # 키워드 검증
            if not dec_type.keywords:
                result.add_info(
                    code="NO_KEYWORDS",
                    message=f"의사결정 타입에 키워드가 없습니다: {name}",
                    path=path
                )

    def _validate_references(
        self,
        ontology: Ontology,
        result: ValidationResult
    ) -> None:
        """참조 무결성 검증"""
        # 고아 관계 타입 검증 (사용되지 않는 관계)
        used_relations: Set[str] = set()

        # 의사결정에서 참조하는 엔티티 타입 수집
        referenced_entities: Set[str] = set()
        for dec_type in ontology.decision_types.values():
            referenced_entities.update(dec_type.relevant_entity_types)

        # 미사용 엔티티 타입 경고
        unused_entities = set(ontology.entity_types.keys()) - referenced_entities
        for unused in unused_entities:
            result.add_info(
                code="UNUSED_ENTITY_TYPE",
                message=f"사용되지 않는 엔티티 타입: {unused}",
                path=f"entity_types.{unused}"
            )

    def _is_valid_identifier(self, name: str) -> bool:
        """유효한 식별자 검사"""
        import re
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))

    def _has_circular_inheritance(
        self,
        entity_name: str,
        ontology: Ontology
    ) -> bool:
        """순환 상속 검사"""
        visited: Set[str] = set()
        current = entity_name

        while current:
            if current in visited:
                return True
            visited.add(current)

            entity_type = ontology.entity_types.get(current)
            if not entity_type:
                break
            current = entity_type.parent

        return False


class EntityValidator:
    """엔티티 데이터 검증기 (온톨로지 기반)"""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def validate_entity(
        self,
        entity_type: str,
        data: Dict[str, Any]
    ) -> ValidationResult:
        """
        엔티티 데이터 검증

        Args:
            entity_type: 엔티티 타입
            data: 엔티티 데이터

        Returns:
            검증 결과
        """
        result = ValidationResult(valid=True)

        # 타입 존재 확인
        type_def = self.ontology.get_entity_type(entity_type)
        if not type_def:
            result.add_error(
                code="UNKNOWN_TYPE",
                message=f"알 수 없는 엔티티 타입: {entity_type}"
            )
            return result

        # 필수 속성 검증
        for req_prop in type_def.required_properties:
            if req_prop not in data or data[req_prop] is None:
                result.add_error(
                    code="MISSING_REQUIRED",
                    message=f"필수 속성 누락: {req_prop}",
                    path=req_prop
                )

        # 속성 타입 검증
        for prop_name, prop_value in data.items():
            if prop_name in type_def.properties:
                expected_type = type_def.properties[prop_name]
                if not self._validate_type(prop_value, expected_type):
                    result.add_warning(
                        code="TYPE_MISMATCH",
                        message=f"타입 불일치: {prop_name} (expected: {expected_type})",
                        path=prop_name
                    )

        return result

    def validate_relation(
        self,
        relation_type: str,
        source_type: str,
        target_type: str
    ) -> ValidationResult:
        """
        관계 검증

        Args:
            relation_type: 관계 타입
            source_type: 소스 엔티티 타입
            target_type: 타겟 엔티티 타입

        Returns:
            검증 결과
        """
        result = ValidationResult(valid=True)

        # 관계 타입 존재 확인
        rel_def = self.ontology.get_relation_type(relation_type)
        if not rel_def:
            result.add_error(
                code="UNKNOWN_RELATION",
                message=f"알 수 없는 관계 타입: {relation_type}"
            )
            return result

        # 소스 타입 검증
        if rel_def.source_types and source_type not in rel_def.source_types:
            result.add_error(
                code="INVALID_SOURCE",
                message=f"유효하지 않은 소스 타입: {source_type}",
                suggestion=f"유효한 소스 타입: {rel_def.source_types}"
            )

        # 타겟 타입 검증
        if rel_def.target_types and target_type not in rel_def.target_types:
            result.add_error(
                code="INVALID_TARGET",
                message=f"유효하지 않은 타겟 타입: {target_type}",
                suggestion=f"유효한 타겟 타입: {rel_def.target_types}"
            )

        return result

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """타입 검증"""
        type_map = {
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        if expected_type.lower() in type_map:
            expected = type_map[expected_type.lower()]
            return isinstance(value, expected)

        # 엔티티 타입 참조인 경우
        if expected_type in self.ontology.entity_types:
            return isinstance(value, (str, dict))

        return True  # 알 수 없는 타입은 통과
