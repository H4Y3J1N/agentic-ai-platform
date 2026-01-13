# Data Architecture Design Document

## 1. Overview

이 문서는 Agentic AI 플랫폼의 데이터 수집, 처리, 저장, 검색을 위한 종합 아키텍처를 정의합니다.

### 핵심 원칙
1. **Ontology-First**: 데이터 수집 전 의미와 관계를 먼저 정의
2. **Decision-Driven**: 의사결정 영향도 기반 데이터 우선순위
3. **Hybrid Search**: 벡터 검색 + 지식 그래프 결합
4. **Adaptive Resolution**: 중요도에 따른 데이터 해상도 조절

---

## 2. Data Schema Layer

### 2.1 Core Domain Models

```
packages/agentic-ai-core/src/agentic_ai_core/
├── schema/
│   ├── __init__.py
│   ├── base.py              # 기본 스키마 클래스
│   ├── document.py          # 문서 스키마
│   ├── entity.py            # 엔티티 스키마
│   ├── relationship.py      # 관계 스키마
│   └── decision.py          # 의사결정 영향 스키마
```

### 2.2 Document Schema

```python
@dataclass
class Document:
    """기본 문서 스키마"""
    id: str
    source_type: SourceType          # notion, slack, confluence, etc.
    document_type: DocumentType      # policy, meeting_note, faq, etc.

    # Content
    raw_content: str
    parsed_content: ParsedContent

    # Metadata (자동 추론 + 수동)
    metadata: DocumentMetadata

    # Decision Relevance
    decision_mappings: List[DecisionMapping]
    relevance_score: float           # 0.0 ~ 1.0

    # Lifecycle
    created_at: datetime
    updated_at: datetime
    resolution_level: ResolutionLevel  # full, summary, metadata_only, archived

    # Graph Relations
    entities: List[EntityRef]
    relationships: List[RelationshipRef]


class DocumentMetadata:
    """문서 메타데이터 (자동 추론 대상)"""
    title: str
    author: Optional[str]
    department: Optional[str]
    topics: List[str]                # 자동 추출
    keywords: List[str]              # 자동 추출
    language: str
    sentiment: Optional[float]
    complexity_score: float          # 문서 복잡도
    freshness_score: float           # 최신성 (decay 적용)
    access_frequency: int            # 조회 빈도
    citation_count: int              # 다른 문서에서 참조된 횟수


class DocumentType(Enum):
    """문서 타입별 파서 매핑"""
    POLICY = "policy"                # 정책/규정 문서
    MEETING_NOTE = "meeting_note"    # 회의록
    FAQ = "faq"                      # FAQ
    TECHNICAL_DOC = "technical_doc"  # 기술 문서
    ANNOUNCEMENT = "announcement"    # 공지사항
    CONVERSATION = "conversation"    # 대화 (Slack)
    WIKI = "wiki"                    # 위키 페이지
    UNKNOWN = "unknown"
```

### 2.3 Entity Schema

```python
@dataclass
class Entity:
    """지식 그래프 엔티티"""
    id: str
    entity_type: EntityType
    name: str
    aliases: List[str]               # 동의어/별칭

    # Properties (타입별 다름)
    properties: Dict[str, Any]

    # Embedding (선택적 벡터화)
    embedding: Optional[List[float]]
    should_vectorize: bool

    # Decision Relevance
    decision_influence: Dict[str, float]  # decision_id -> influence_score

    # Provenance
    source_documents: List[str]      # 추출된 문서 ID들
    confidence: float                # 추출 신뢰도


class EntityType(Enum):
    PERSON = "person"
    TEAM = "team"
    PROJECT = "project"
    PRODUCT = "product"
    PROCESS = "process"
    POLICY = "policy"
    TOOL = "tool"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"
```

### 2.4 Relationship Schema

```python
@dataclass
class Relationship:
    """엔티티 간 관계"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType

    # Properties
    properties: Dict[str, Any]
    weight: float                    # 관계 강도

    # Temporal
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]

    # Provenance
    source_documents: List[str]
    confidence: float


class RelationType(Enum):
    # 조직 관계
    BELONGS_TO = "belongs_to"        # 소속
    MANAGES = "manages"              # 관리
    REPORTS_TO = "reports_to"        # 보고

    # 프로젝트 관계
    WORKS_ON = "works_on"
    OWNS = "owns"
    CONTRIBUTES_TO = "contributes_to"

    # 문서 관계
    REFERENCES = "references"
    SUPERSEDES = "supersedes"        # 대체
    RELATED_TO = "related_to"

    # 프로세스 관계
    DEPENDS_ON = "depends_on"
    TRIGGERS = "triggers"
    PART_OF = "part_of"
```

---

## 3. Ontology Layer

### 3.1 도메인 온톨로지 구조

```
config/ontology/
├── base_ontology.yaml       # 공통 온톨로지
├── internal_ops.yaml        # internal-ops 도메인
└── schemas/
    ├── entity_types.yaml
    ├── relation_types.yaml
    └── decision_types.yaml
```

### 3.2 온톨로지 정의 예시

```yaml
# config/ontology/internal_ops.yaml
ontology:
  name: "internal-ops"
  version: "1.0.0"

  # 엔티티 타입 정의
  entity_types:
    employee:
      properties:
        - name: "name"
          type: "string"
          required: true
        - name: "department"
          type: "reference"
          ref: "department"
        - name: "role"
          type: "string"
      vectorize: false  # 엔티티 자체는 벡터화 안함

    department:
      properties:
        - name: "name"
          type: "string"
        - name: "parent"
          type: "reference"
          ref: "department"
      vectorize: false

    policy:
      properties:
        - name: "title"
          type: "string"
        - name: "category"
          type: "enum"
          values: ["hr", "finance", "security", "operations"]
        - name: "effective_date"
          type: "date"
      vectorize: true  # 정책 내용은 벡터화

  # 관계 타입 정의
  relation_types:
    employee_department:
      source: "employee"
      target: "department"
      relation: "belongs_to"
      cardinality: "many_to_one"

    policy_department:
      source: "policy"
      target: "department"
      relation: "applies_to"
      cardinality: "many_to_many"

  # 의사결정 타입 정의
  decision_types:
    leave_request:
      description: "휴가 신청 관련 의사결정"
      relevant_entities: ["employee", "policy", "department"]
      relevant_document_types: ["policy", "faq"]

    onboarding:
      description: "신규 입사자 온보딩 관련"
      relevant_entities: ["employee", "department", "process"]
      relevant_document_types: ["policy", "technical_doc", "wiki"]
```

---

## 4. Decision Influence Mapping

### 4.1 의사결정 영향도 스키마

```python
@dataclass
class DecisionType:
    """의사결정 유형 정의"""
    id: str
    name: str
    description: str
    frequency: DecisionFrequency     # daily, weekly, monthly, rare
    impact_level: ImpactLevel        # low, medium, high, critical

    # 관련 데이터
    relevant_entity_types: List[EntityType]
    relevant_document_types: List[DocumentType]

    # 가중치 학습
    learned_weights: Dict[str, float]  # data_point_id -> weight


@dataclass
class DecisionMapping:
    """문서/엔티티 → 의사결정 매핑"""
    decision_type_id: str
    influence_score: float           # 0.0 ~ 1.0
    influence_type: InfluenceType    # direct, indirect, contextual

    # 학습된 메트릭
    query_hit_count: int             # 해당 의사결정 쿼리에서 검색된 횟수
    feedback_score: float            # 사용자 피드백 기반 점수

    # 계산 근거
    reasoning: str                   # 왜 이 점수인지


class InfluenceType(Enum):
    DIRECT = "direct"                # 직접적 영향 (정책 → 의사결정)
    INDIRECT = "indirect"            # 간접적 영향 (참조 문서)
    CONTEXTUAL = "contextual"        # 맥락 제공
```

### 4.2 영향도 계산 공식

```python
def calculate_decision_relevance(
    document: Document,
    decision_type: DecisionType
) -> float:
    """
    의사결정 관련성 점수 계산

    Score = Σ(weight_i × factor_i) / Σ(weight_i)
    """
    factors = {
        # 정적 요소
        "document_type_match": (0.3, 1.0 if doc.type in decision.relevant_types else 0.0),
        "entity_overlap": (0.2, calculate_entity_overlap(doc, decision)),
        "keyword_match": (0.1, calculate_keyword_similarity(doc, decision)),

        # 동적 요소 (학습됨)
        "query_hit_rate": (0.2, doc.decision_mappings.get(decision.id, {}).get("hit_rate", 0)),
        "user_feedback": (0.15, doc.decision_mappings.get(decision.id, {}).get("feedback", 0.5)),

        # 시간 요소
        "freshness": (0.05, calculate_freshness(doc.updated_at)),
    }

    weighted_sum = sum(weight * score for weight, score in factors.values())
    total_weight = sum(weight for weight, _ in factors.values())

    return weighted_sum / total_weight
```

---

## 5. Data Ingestion Pipeline

### 5.1 파이프라인 구조

```
packages/agentic-ai-core/src/agentic_ai_core/
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py          # 메인 파이프라인
│   ├── stages/
│   │   ├── fetch.py         # 데이터 수집
│   │   ├── parse.py         # 파싱 (타입별)
│   │   ├── extract.py       # 엔티티/관계 추출
│   │   ├── infer.py         # 메타데이터 추론
│   │   ├── score.py         # 관련성 스코어링
│   │   ├── vectorize.py     # 선택적 벡터화
│   │   └── store.py         # 저장
│   ├── parsers/
│   │   ├── base.py
│   │   ├── notion_parser.py
│   │   ├── slack_parser.py
│   │   ├── markdown_parser.py
│   │   └── pdf_parser.py
│   └── extractors/
│       ├── entity_extractor.py
│       ├── relation_extractor.py
│       └── metadata_inferrer.py
```

### 5.2 파이프라인 정의

```python
class IngestionPipeline:
    """데이터 수집 파이프라인"""

    def __init__(self, config: PipelineConfig):
        self.stages = [
            FetchStage(config.sources),
            ParseStage(config.parsers),
            ExtractStage(config.extractors),
            InferStage(config.inference_models),
            ScoreStage(config.decision_types),
            VectorizeStage(config.vectorization_rules),
            StoreStage(config.stores),
        ]

    async def process(self, source_item: SourceItem) -> ProcessedDocument:
        """단일 아이템 처리"""
        context = PipelineContext(source_item)

        for stage in self.stages:
            try:
                context = await stage.process(context)

                # 조기 종료 조건
                if context.should_skip:
                    logger.info(f"Skipping {source_item.id}: {context.skip_reason}")
                    break

            except StageError as e:
                context.errors.append(e)
                if e.is_fatal:
                    raise

        return context.to_document()


@dataclass
class PipelineContext:
    """파이프라인 컨텍스트 (스테이지 간 데이터 전달)"""
    source_item: SourceItem

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
    vectorization_decision: VectorizationDecision = None
    embeddings: Dict[str, List[float]] = field(default_factory=dict)

    # 제어
    should_skip: bool = False
    skip_reason: Optional[str] = None
    errors: List[StageError] = field(default_factory=list)
```

### 5.3 문서 타입별 파서

```python
class ParserRegistry:
    """문서 타입별 파서 레지스트리"""

    _parsers: Dict[SourceType, Dict[DocumentType, BaseParser]] = {}

    @classmethod
    def register(cls, source_type: SourceType, doc_type: DocumentType):
        def decorator(parser_cls):
            if source_type not in cls._parsers:
                cls._parsers[source_type] = {}
            cls._parsers[source_type][doc_type] = parser_cls()
            return parser_cls
        return decorator

    @classmethod
    def get_parser(cls, source_type: SourceType, doc_type: DocumentType) -> BaseParser:
        return cls._parsers.get(source_type, {}).get(doc_type, DefaultParser())


# 파서 등록 예시
@ParserRegistry.register(SourceType.NOTION, DocumentType.MEETING_NOTE)
class NotionMeetingNoteParser(BaseParser):
    """Notion 회의록 전용 파서"""

    def parse(self, content: str, metadata: Dict) -> ParsedContent:
        sections = {
            "attendees": self._extract_attendees(content),
            "agenda": self._extract_agenda(content),
            "decisions": self._extract_decisions(content),
            "action_items": self._extract_action_items(content),
        }
        return ParsedContent(
            document_type=DocumentType.MEETING_NOTE,
            sections=sections,
            structured_data=self._to_structured(sections)
        )


@ParserRegistry.register(SourceType.SLACK, DocumentType.CONVERSATION)
class SlackConversationParser(BaseParser):
    """Slack 대화 파서"""

    def parse(self, messages: List[Dict], metadata: Dict) -> ParsedContent:
        # 대화 스레드 구조화
        threads = self._group_threads(messages)

        # 중요 메시지 식별
        important = self._identify_important(messages)

        return ParsedContent(
            document_type=DocumentType.CONVERSATION,
            sections={"threads": threads, "highlights": important},
            structured_data=self._extract_decisions_and_actions(threads)
        )
```

---

## 6. Metadata Inference

### 6.1 자동 추론 파이프라인

```python
class MetadataInferrer:
    """메타데이터 자동 추론"""

    def __init__(self, llm_client, ontology: Ontology):
        self.llm = llm_client
        self.ontology = ontology
        self.extractors = [
            TopicExtractor(),
            KeywordExtractor(),
            SentimentAnalyzer(),
            ComplexityScorer(),
            DocumentTypeClassifier(ontology),
            DepartmentInferrer(ontology),
        ]

    async def infer(self, content: str, existing_metadata: Dict) -> InferredMetadata:
        """메타데이터 추론"""
        results = {}

        # 병렬 추출
        tasks = [ext.extract(content) for ext in self.extractors]
        extracted = await asyncio.gather(*tasks)

        for extractor, result in zip(self.extractors, extracted):
            results[extractor.field_name] = result

        # LLM 기반 고급 추론 (필요시)
        if self._needs_llm_inference(results, existing_metadata):
            llm_inferred = await self._llm_infer(content, results)
            results.update(llm_inferred)

        return InferredMetadata(**results)

    async def _llm_infer(self, content: str, partial_results: Dict) -> Dict:
        """LLM 기반 추론 (비용 고려하여 선택적 사용)"""
        prompt = f"""
        다음 문서를 분석하여 메타데이터를 추론하세요.

        문서 내용 (일부):
        {content[:2000]}

        이미 추출된 정보:
        {json.dumps(partial_results, ensure_ascii=False)}

        추론할 항목:
        1. 이 문서가 어떤 부서/팀과 관련있는지
        2. 어떤 의사결정에 영향을 주는지
        3. 문서의 주요 대상 독자

        JSON 형식으로 응답:
        """

        response = await self.llm.complete(prompt)
        return json.loads(response)
```

### 6.2 엔티티 추출

```python
class EntityExtractor:
    """엔티티 자동 추출"""

    def __init__(self, ontology: Ontology, ner_model, llm_client):
        self.ontology = ontology
        self.ner = ner_model
        self.llm = llm_client

        # 기존 엔티티 캐시 (중복 방지)
        self.entity_cache: Dict[str, Entity] = {}

    async def extract(self, content: str, context: PipelineContext) -> List[Entity]:
        """엔티티 추출"""
        entities = []

        # 1. NER 기반 추출
        ner_entities = self.ner.extract(content)

        # 2. 온톨로지 기반 패턴 매칭
        pattern_entities = self._pattern_match(content)

        # 3. 기존 엔티티와 매칭 (중복 제거)
        merged = self._merge_and_dedupe(ner_entities + pattern_entities)

        # 4. 신규 엔티티는 LLM으로 속성 추론
        for entity in merged:
            if entity.id not in self.entity_cache:
                entity.properties = await self._infer_properties(entity, content)
                entities.append(entity)
            else:
                # 기존 엔티티 참조
                entities.append(self.entity_cache[entity.id])

        return entities

    def _pattern_match(self, content: str) -> List[Entity]:
        """온톨로지 정의 기반 패턴 매칭"""
        entities = []

        for entity_type in self.ontology.entity_types:
            patterns = entity_type.extraction_patterns
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    entities.append(Entity(
                        id=self._generate_id(entity_type, match),
                        entity_type=entity_type,
                        name=match,
                        confidence=0.8  # 패턴 매칭 신뢰도
                    ))

        return entities
```

---

## 7. Knowledge Graph Storage

### 7.1 그래프 스토어 인터페이스

```python
class KnowledgeGraphStore(ABC):
    """지식 그래프 저장소 인터페이스"""

    @abstractmethod
    async def add_entity(self, entity: Entity) -> str:
        """엔티티 추가"""
        pass

    @abstractmethod
    async def add_relationship(self, rel: Relationship) -> str:
        """관계 추가"""
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """엔티티 조회"""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        depth: int = 1
    ) -> List[Entity]:
        """이웃 엔티티 조회"""
        pass

    @abstractmethod
    async def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Cypher 쿼리 실행"""
        pass

    @abstractmethod
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> Optional[List[Entity]]:
        """두 엔티티 간 경로 찾기"""
        pass


class Neo4jKnowledgeGraph(KnowledgeGraphStore):
    """Neo4j 기반 구현"""

    async def get_neighbors(self, entity_id: str, relation_types=None, depth=1):
        cypher = """
        MATCH (e:Entity {id: $entity_id})-[r*1..%d]-(neighbor:Entity)
        WHERE $relation_types IS NULL OR type(r) IN $relation_types
        RETURN DISTINCT neighbor
        """ % depth

        return await self.query(cypher, {
            "entity_id": entity_id,
            "relation_types": relation_types
        })


class SQLiteKnowledgeGraph(KnowledgeGraphStore):
    """경량 SQLite 기반 구현 (개발/소규모용)"""
    pass
```

### 7.2 하이브리드 검색 (벡터 + 그래프)

```python
class HybridSearchEngine:
    """벡터 검색 + 지식 그래프 결합"""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: KnowledgeGraphStore,
        reranker: Optional[Reranker] = None
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_mode: SearchMode = SearchMode.HYBRID,
        decision_context: Optional[DecisionType] = None
    ) -> List[SearchResult]:
        """하이브리드 검색"""

        results = []

        # 1. 벡터 검색
        if search_mode in [SearchMode.VECTOR, SearchMode.HYBRID]:
            vector_results = await self._vector_search(query, top_k * 2)
            results.extend(vector_results)

        # 2. 그래프 검색 (쿼리에서 엔티티 추출 후)
        if search_mode in [SearchMode.GRAPH, SearchMode.HYBRID]:
            query_entities = await self._extract_query_entities(query)
            graph_results = await self._graph_search(query_entities, top_k * 2)
            results.extend(graph_results)

        # 3. 결과 병합 및 중복 제거
        merged = self._merge_results(results)

        # 4. 의사결정 컨텍스트 기반 재순위
        if decision_context:
            merged = self._apply_decision_boost(merged, decision_context)

        # 5. Reranker 적용 (선택적)
        if self.reranker:
            merged = await self.reranker.rerank(query, merged)

        return merged[:top_k]

    async def _graph_search(
        self,
        entities: List[Entity],
        top_k: int
    ) -> List[SearchResult]:
        """그래프 기반 검색"""
        results = []

        for entity in entities:
            # 연결된 문서 찾기
            docs = await self.graph_store.query("""
                MATCH (e:Entity {id: $entity_id})-[:MENTIONED_IN]->(d:Document)
                RETURN d, count(*) as mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
            """, {"entity_id": entity.id, "limit": top_k})

            # 관련 엔티티 경유 문서
            related_docs = await self.graph_store.query("""
                MATCH (e:Entity {id: $entity_id})-[*1..2]-(related:Entity)
                      -[:MENTIONED_IN]->(d:Document)
                RETURN d, count(DISTINCT related) as connection_strength
                ORDER BY connection_strength DESC
                LIMIT $limit
            """, {"entity_id": entity.id, "limit": top_k})

            results.extend(docs + related_docs)

        return results

    def _apply_decision_boost(
        self,
        results: List[SearchResult],
        decision: DecisionType
    ) -> List[SearchResult]:
        """의사결정 관련성에 따른 점수 부스트"""
        for result in results:
            mapping = result.document.decision_mappings.get(decision.id)
            if mapping:
                boost = 1.0 + (mapping.influence_score * 0.5)  # 최대 50% 부스트
                result.score *= boost

        return sorted(results, key=lambda r: r.score, reverse=True)
```

---

## 8. Vectorization Strategy

### 8.1 벡터화 결정 로직

```python
@dataclass
class VectorizationDecision:
    """벡터화 결정"""
    should_vectorize: bool
    reason: str
    granularity: VectorizationGranularity  # full, chunks, summary, none
    priority: int                           # 1-10, 높을수록 우선


class VectorizationGranularity(Enum):
    FULL = "full"           # 전체 문서 벡터화
    CHUNKS = "chunks"       # 청크 단위 벡터화
    SUMMARY = "summary"     # 요약만 벡터화
    METADATA = "metadata"   # 메타데이터만 벡터화
    NONE = "none"           # 벡터화 안함


class VectorizationDecider:
    """벡터화 결정 로직"""

    def __init__(self, config: VectorizationConfig):
        self.config = config

    def decide(
        self,
        document: Document,
        decision_mappings: List[DecisionMapping]
    ) -> VectorizationDecision:
        """벡터화 여부 및 방식 결정"""

        # 규칙 기반 결정
        rules_decision = self._apply_rules(document)
        if rules_decision.should_vectorize is False:
            return rules_decision

        # 의사결정 영향도 기반
        max_influence = max(
            (m.influence_score for m in decision_mappings),
            default=0.0
        )

        # 벡터화 ROI 계산
        roi = self._calculate_vectorization_roi(document, max_influence)

        if roi < self.config.min_roi_threshold:
            return VectorizationDecision(
                should_vectorize=False,
                reason=f"Low ROI: {roi:.2f}",
                granularity=VectorizationGranularity.NONE,
                priority=0
            )

        # 세분화 수준 결정
        granularity = self._decide_granularity(document, max_influence)

        return VectorizationDecision(
            should_vectorize=True,
            reason=f"ROI: {roi:.2f}, Influence: {max_influence:.2f}",
            granularity=granularity,
            priority=int(max_influence * 10)
        )

    def _calculate_vectorization_roi(
        self,
        document: Document,
        decision_influence: float
    ) -> float:
        """
        벡터화 ROI 계산

        ROI = (Expected_Query_Value × Query_Probability) / Storage_Cost
        """
        # 예상 쿼리 가치 (의사결정 영향도 기반)
        query_value = decision_influence * self.config.base_query_value

        # 쿼리 확률 (문서 타입, 최신성 기반)
        query_prob = self._estimate_query_probability(document)

        # 저장 비용 (문서 크기, 청크 수)
        storage_cost = self._estimate_storage_cost(document)

        return (query_value * query_prob) / max(storage_cost, 0.01)

    def _decide_granularity(
        self,
        document: Document,
        influence: float
    ) -> VectorizationGranularity:
        """세분화 수준 결정"""
        content_length = len(document.raw_content)

        # 짧은 문서 → 전체
        if content_length < 1000:
            return VectorizationGranularity.FULL

        # 높은 영향도 → 청크 단위 (정밀 검색)
        if influence > 0.7:
            return VectorizationGranularity.CHUNKS

        # 중간 영향도 → 요약
        if influence > 0.3:
            return VectorizationGranularity.SUMMARY

        # 낮은 영향도 → 메타데이터만
        return VectorizationGranularity.METADATA
```

### 8.2 청킹 전략

```python
class SmartChunker:
    """지능형 청킹"""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(
        self,
        document: Document,
        granularity: VectorizationGranularity
    ) -> List[Chunk]:
        """문서 청킹"""

        if granularity == VectorizationGranularity.FULL:
            return [Chunk(content=document.raw_content, metadata=document.metadata)]

        if granularity == VectorizationGranularity.SUMMARY:
            summary = self._generate_summary(document)
            return [Chunk(content=summary, metadata=document.metadata)]

        if granularity == VectorizationGranularity.METADATA:
            return [Chunk(
                content=self._metadata_to_text(document.metadata),
                metadata=document.metadata
            )]

        # CHUNKS: 문서 타입별 전략
        strategy = self._get_chunking_strategy(document.document_type)
        return strategy.chunk(document)

    def _get_chunking_strategy(self, doc_type: DocumentType) -> ChunkingStrategy:
        """문서 타입별 청킹 전략"""
        strategies = {
            DocumentType.POLICY: SemanticChunker(
                min_chunk_size=200,
                max_chunk_size=800,
                respect_headers=True
            ),
            DocumentType.MEETING_NOTE: SectionChunker(
                sections=["attendees", "agenda", "decisions", "action_items"]
            ),
            DocumentType.CONVERSATION: ThreadChunker(
                max_messages_per_chunk=10,
                preserve_context=True
            ),
            DocumentType.TECHNICAL_DOC: CodeAwareChunker(
                preserve_code_blocks=True,
                max_chunk_size=1000
            ),
        }
        return strategies.get(doc_type, DefaultChunker())
```

---

## 9. Data Lifecycle Management

### 9.1 해상도 관리

```python
class ResolutionLevel(Enum):
    FULL = "full"                # 전체 데이터 유지
    SUMMARY = "summary"          # 요약만 유지
    METADATA_ONLY = "metadata"   # 메타데이터만 유지
    ARCHIVED = "archived"        # 아카이브 (저비용 스토리지)
    DELETED = "deleted"          # 삭제됨


class DataLifecycleManager:
    """데이터 생명주기 관리"""

    def __init__(self, config: LifecycleConfig):
        self.config = config

    async def evaluate_and_adjust(self, document: Document) -> ResolutionLevel:
        """문서 해상도 재평가 및 조정"""

        # 현재 메트릭 수집
        metrics = await self._collect_metrics(document)

        # 새 해상도 결정
        new_resolution = self._decide_resolution(document, metrics)

        if new_resolution != document.resolution_level:
            await self._apply_resolution_change(document, new_resolution)

        return new_resolution

    def _decide_resolution(
        self,
        document: Document,
        metrics: DocumentMetrics
    ) -> ResolutionLevel:
        """해상도 결정"""

        # 강제 유지 조건
        if document.document_type in self.config.always_keep_types:
            return ResolutionLevel.FULL

        # 점수 계산
        retention_score = self._calculate_retention_score(document, metrics)

        if retention_score > 0.8:
            return ResolutionLevel.FULL
        elif retention_score > 0.5:
            return ResolutionLevel.SUMMARY
        elif retention_score > 0.2:
            return ResolutionLevel.METADATA_ONLY
        elif retention_score > 0.05:
            return ResolutionLevel.ARCHIVED
        else:
            return ResolutionLevel.DELETED

    def _calculate_retention_score(
        self,
        document: Document,
        metrics: DocumentMetrics
    ) -> float:
        """
        유지 점수 계산

        Score = w1×DecisionInfluence + w2×AccessFrequency + w3×Freshness + w4×CitationCount
        """
        weights = self.config.retention_weights

        scores = {
            "decision_influence": max(
                (m.influence_score for m in document.decision_mappings),
                default=0.0
            ),
            "access_frequency": min(metrics.access_count / 100, 1.0),
            "freshness": self._calculate_freshness(document.updated_at),
            "citation_count": min(metrics.citation_count / 10, 1.0),
        }

        return sum(weights[k] * scores[k] for k in scores)

    async def run_lifecycle_job(self):
        """주기적 생명주기 관리 작업"""
        documents = await self._get_documents_for_review()

        for doc in documents:
            try:
                new_resolution = await self.evaluate_and_adjust(doc)
                logger.info(f"Document {doc.id}: {doc.resolution_level} -> {new_resolution}")
            except Exception as e:
                logger.error(f"Failed to process {doc.id}: {e}")
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Core schema 정의 (`schema/`)
- [ ] 온톨로지 YAML 구조 설계
- [ ] 기본 파이프라인 프레임워크

### Phase 2: Ingestion (Week 3-4)
- [ ] 문서 타입별 파서 구현
- [ ] 메타데이터 추론 파이프라인
- [ ] 엔티티/관계 추출기

### Phase 3: Knowledge Graph (Week 5-6)
- [ ] 그래프 스토어 인터페이스
- [ ] SQLite 경량 구현 (개발용)
- [ ] Neo4j 연동 (프로덕션용)

### Phase 4: Hybrid Search (Week 7-8)
- [ ] 벡터 + 그래프 통합 검색
- [ ] 의사결정 기반 재순위
- [ ] 캐싱 레이어

### Phase 5: Lifecycle (Week 9-10)
- [ ] 해상도 관리 시스템
- [ ] 벡터화 ROI 계산
- [ ] 자동 정리 작업

---

## 11. 디렉토리 구조 (최종)

```
packages/agentic-ai-core/src/agentic_ai_core/
├── schema/
│   ├── __init__.py
│   ├── base.py
│   ├── document.py
│   ├── entity.py
│   ├── relationship.py
│   └── decision.py
├── ontology/
│   ├── __init__.py
│   ├── loader.py
│   └── validator.py
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── stages/
│   ├── parsers/
│   └── extractors/
├── graph/
│   ├── __init__.py
│   ├── store.py
│   ├── neo4j_store.py
│   └── sqlite_store.py
├── search/
│   ├── __init__.py
│   ├── hybrid_engine.py
│   ├── vector_search.py
│   └── graph_search.py
├── vectorization/
│   ├── __init__.py
│   ├── decider.py
│   ├── chunker.py
│   └── embedder.py
├── lifecycle/
│   ├── __init__.py
│   ├── manager.py
│   └── resolution.py
└── scoring/
    ├── __init__.py
    ├── decision_scorer.py
    └── roi_calculator.py
```
