"""
Extract Stage

엔티티 및 관계 추출 스테이지
"""

from typing import Optional, List, Dict, Any, Set
import re
import logging

from .base import Stage
from ..context import PipelineContext
from ...schema import Entity, EntityType, Relationship, RelationType

logger = logging.getLogger(__name__)


class ExtractStage(Stage):
    """엔티티/관계 추출 스테이지"""

    def __init__(
        self,
        entity_patterns: Optional[Dict[EntityType, List[str]]] = None,
        use_llm: bool = False
    ):
        super().__init__("ExtractStage")
        self.entity_patterns = entity_patterns or DEFAULT_ENTITY_PATTERNS
        self.use_llm = use_llm
        self._entity_cache: Dict[str, Entity] = {}

    async def process(self, context: PipelineContext) -> PipelineContext:
        """추출 실행"""
        content = context.raw_content or ""
        if not content:
            return context

        # 엔티티 추출
        entities = await self._extract_entities(content, context)
        context.entities = entities

        # 관계 추출
        relationships = await self._extract_relationships(content, entities, context)
        context.relationships = relationships

        logger.debug(
            f"Extracted {len(entities)} entities, {len(relationships)} relationships"
        )

        return context

    async def _extract_entities(
        self,
        content: str,
        context: PipelineContext
    ) -> List[Entity]:
        """엔티티 추출"""
        entities = []
        seen_names: Set[str] = set()

        # 패턴 기반 추출
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    name = match.group(1) if match.groups() else match.group(0)
                    name = name.strip()

                    if not name or name.lower() in seen_names:
                        continue

                    seen_names.add(name.lower())

                    # 캐시 확인
                    cache_key = f"{entity_type.value}:{name.lower()}"
                    if cache_key in self._entity_cache:
                        entities.append(self._entity_cache[cache_key])
                        continue

                    entity = Entity(
                        entity_type=entity_type,
                        name=name,
                        confidence=0.7,  # 패턴 매칭 신뢰도
                    )
                    entity.add_source(context.source_item.id)
                    entities.append(entity)
                    self._entity_cache[cache_key] = entity

        # 이메일 추출
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        for match in re.finditer(email_pattern, content):
            email = match.group(0)
            if email.lower() not in seen_names:
                seen_names.add(email.lower())
                entity = Entity(
                    entity_type=EntityType.PERSON,
                    name=email.split("@")[0],
                    properties={"email": email},
                    confidence=0.9,
                )
                entity.add_source(context.source_item.id)
                entities.append(entity)

        return entities

    async def _extract_relationships(
        self,
        content: str,
        entities: List[Entity],
        context: PipelineContext
    ) -> List[Relationship]:
        """관계 추출"""
        relationships = []

        # 간단한 공동 출현 기반 관계
        content_lower = content.lower()

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1:]:
                # 같은 문장에 등장하면 관련 관계
                sentences = content.split(".")
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if (
                        entity1.normalized_name in sentence_lower and
                        entity2.normalized_name in sentence_lower
                    ):
                        rel = Relationship(
                            relation_type=RelationType.RELATED_TO,
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            confidence=0.5,
                        )
                        rel.add_source(context.source_item.id)
                        relationships.append(rel)
                        break

        # 키워드 기반 관계 추출
        relation_keywords = {
            RelationType.MANAGES: ["manages", "관리", "담당"],
            RelationType.BELONGS_TO: ["belongs to", "소속", "부서"],
            RelationType.WORKS_ON: ["works on", "참여", "작업"],
            RelationType.OWNS: ["owns", "소유", "책임"],
        }

        for rel_type, keywords in relation_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # 키워드 주변 엔티티 찾기
                    keyword_pos = content_lower.find(keyword)
                    context_start = max(0, keyword_pos - 100)
                    context_end = min(len(content), keyword_pos + 100)
                    context_text = content_lower[context_start:context_end]

                    nearby_entities = [
                        e for e in entities
                        if e.normalized_name in context_text
                    ]

                    if len(nearby_entities) >= 2:
                        rel = Relationship(
                            relation_type=rel_type,
                            source_entity_id=nearby_entities[0].id,
                            target_entity_id=nearby_entities[1].id,
                            confidence=0.6,
                        )
                        rel.add_source(context.source_item.id)
                        relationships.append(rel)

        return relationships


# 기본 엔티티 패턴
DEFAULT_ENTITY_PATTERNS: Dict[EntityType, List[str]] = {
    EntityType.PERSON: [
        r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([가-힣]{2,4})\s*(?:님|씨|대리|과장|부장|팀장|사원)',
    ],
    EntityType.TEAM: [
        r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:team|팀)',
        r'([가-힣]+)\s*팀',
    ],
    EntityType.DEPARTMENT: [
        r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:department|부서)',
        r'([가-힣]+)\s*(?:부|부서|본부)',
    ],
    EntityType.PROJECT: [
        r'(?:project|프로젝트)\s+["\']?([A-Za-z0-9가-힣\s]+)["\']?',
        r'([A-Za-z0-9가-힣]+)\s+(?:project|프로젝트)',
    ],
    EntityType.POLICY: [
        r'([가-힣A-Za-z]+)\s*(?:정책|규정|지침)',
        r'(?:policy|regulation):\s*([A-Za-z\s]+)',
    ],
    EntityType.TOOL: [
        r'(?:using|사용)\s+([A-Za-z0-9]+)',
    ],
}
