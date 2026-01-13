# Implementation Strategy: Core Package Structure

## 1. 최종 폴더 구조

```
packages/agentic-ai-core/src/agentic_ai_core/
│
├── __init__.py
│
├── schema/                      # Phase 1: 데이터 스키마
│   ├── __init__.py
│   ├── base.py                  # 공통 베이스 클래스
│   ├── document.py              # Document, Chunk, ParsedContent
│   ├── entity.py                # Entity, EntityType
│   ├── relationship.py          # Relationship, RelationType
│   └── decision.py              # DecisionType, DecisionMapping
│
├── rag/                         # Phase 2: 벡터 검색 (범위: 벡터만)
│   ├── __init__.py
│   ├── chunker.py               # 청킹 전략
│   ├── embedder.py              # 임베딩 생성
│   ├── retriever.py             # 벡터 검색
│   ├── indexer.py               # 인덱싱 관리
│   └── stores/
│       ├── __init__.py
│       ├── base.py              # VectorStore ABC
│       └── chroma_store.py      # ChromaDB 구현
│
├── ingestion/                   # Phase 3: 데이터 수집
│   ├── __init__.py
│   ├── pipeline.py              # 메인 파이프라인
│   ├── context.py               # PipelineContext
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py              # Stage ABC
│   │   ├── fetch.py             # 데이터 가져오기
│   │   ├── parse.py             # 파싱
│   │   ├── extract.py           # 엔티티/관계 추출
│   │   ├── infer.py             # 메타데이터 추론
│   │   ├── score.py             # 관련성 스코어링
│   │   ├── vectorize.py         # 벡터화 결정 및 실행
│   │   └── store.py             # 저장
│   └── parsers/
│       ├── __init__.py
│       ├── base.py              # Parser ABC
│       ├── notion_parser.py
│       ├── slack_parser.py
│       └── markdown_parser.py
│
├── graph/                       # Phase 4: 지식 그래프
│   ├── __init__.py
│   ├── store.py                 # GraphStore ABC
│   ├── sqlite_store.py          # 경량 SQLite 구현
│   └── queries.py               # 공통 쿼리 헬퍼
│
├── search/                      # Phase 5: 하이브리드 검색
│   ├── __init__.py
│   ├── engine.py                # HybridSearchEngine
│   ├── reranker.py              # 재순위 로직
│   └── filters.py               # 검색 필터
│
├── scoring/                     # Phase 6: 스코어링
│   ├── __init__.py
│   ├── decision_scorer.py       # 의사결정 영향도
│   ├── relevance_scorer.py      # 검색 관련성
│   └── roi_calculator.py        # 벡터화 ROI
│
├── ontology/                    # Phase 7: 온톨로지
│   ├── __init__.py
│   ├── loader.py                # YAML 로더
│   └── validator.py             # 스키마 검증
│
├── lifecycle/                   # Phase 8: 생명주기
│   ├── __init__.py
│   ├── manager.py               # 해상도 관리
│   └── scheduler.py             # 정리 작업 스케줄러
│
└── api/                         # 기존 유지
    ├── __init__.py
    ├── sse_response.py
    └── websocket_manager.py
```

---

## 2. 구현 순서 및 의존성

```
Phase 1: schema/
    ↓ (schema 정의 완료)
Phase 2: rag/
    ↓ (벡터 검색 가능)
Phase 3: ingestion/
    ↓ (데이터 수집 가능)
Phase 4: graph/
    ↓ (그래프 저장 가능)
Phase 5: search/
    ↓ (하이브리드 검색 가능)
Phase 6: scoring/
    ↓ (스코어링 가능)
Phase 7: ontology/
    ↓ (온톨로지 로드 가능)
Phase 8: lifecycle/
    (전체 시스템 완성)
```

### 의존성 다이어그램

```
schema ──────────────────────────────────────┐
   │                                         │
   ├──→ rag (chunker uses Document)          │
   │      │                                  │
   │      └──→ ingestion (uses rag)          │
   │             │                           │
   ├──→ graph (uses Entity, Relationship) ←──┤
   │      │                                  │
   │      └──→ search (uses rag + graph)     │
   │             │                           │
   ├──→ scoring (uses DecisionMapping) ←─────┤
   │             │                           │
   └──→ lifecycle (uses all above)           │
                 │                           │
         ontology (standalone, used by all) ─┘
```

---

## 3. Phase 1: schema/ 상세

### 3.1 파일별 내용

| 파일 | 클래스 | 설명 |
|------|--------|------|
| `base.py` | `BaseModel`, `Identifiable`, `Timestamped` | 공통 믹스인 |
| `document.py` | `Document`, `Chunk`, `ParsedContent`, `DocumentMetadata`, `DocumentType`, `SourceType` | 문서 관련 |
| `entity.py` | `Entity`, `EntityType`, `EntityRef` | 엔티티 관련 |
| `relationship.py` | `Relationship`, `RelationType`, `RelationshipRef` | 관계 관련 |
| `decision.py` | `DecisionType`, `DecisionMapping`, `InfluenceType` | 의사결정 관련 |

### 3.2 핵심 설계 원칙

```python
# 1. Pydantic 기반 (검증 + 직렬화)
from pydantic import BaseModel, Field

# 2. 불변성 권장
class Document(BaseModel):
    class Config:
        frozen = True  # 불변 객체

# 3. 타입 힌트 완전 적용
def process(doc: Document) -> ProcessedDocument:
    ...

# 4. 팩토리 메서드 패턴
class Entity:
    @classmethod
    def from_notion(cls, data: dict) -> "Entity":
        ...

# 5. 직렬화 지원
    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        ...
```

---

## 4. Phase 2: rag/ 상세

### 4.1 파일별 내용

| 파일 | 클래스 | 의존성 |
|------|--------|--------|
| `chunker.py` | `Chunker`, `ChunkingStrategy`, `SemanticChunker` | schema.Document |
| `embedder.py` | `Embedder`, `OpenAIEmbedder` | - |
| `retriever.py` | `Retriever` | stores, schema.Document |
| `indexer.py` | `Indexer` | chunker, embedder, stores |
| `stores/base.py` | `VectorStore` (ABC) | schema.Chunk |
| `stores/chroma_store.py` | `ChromaStore` | VectorStore |

### 4.2 stores/ 분리 이유

```
rag/
├── chroma_store.py    # 현재
└── stores/            # 변경 후
    ├── base.py        # ABC
    └── chroma_store.py
```

- **확장성**: 나중에 Pinecone, Weaviate 등 추가 시 일관된 구조
- **테스트**: Mock Store 쉽게 주입 가능
- **의존성 역전**: Retriever는 ABC에만 의존

---

## 5. Phase 3: ingestion/ 상세

### 5.1 파이프라인 구조

```
[SourceItem]
     │
     ▼
┌─────────┐
│  Fetch  │  → raw data 획득
└────┬────┘
     ▼
┌─────────┐
│  Parse  │  → ParsedContent 생성 (타입별 파서)
└────┬────┘
     ▼
┌─────────┐
│ Extract │  → Entity, Relationship 추출
└────┬────┘
     ▼
┌─────────┐
│  Infer  │  → 메타데이터 자동 추론
└────┬────┘
     ▼
┌─────────┐
│  Score  │  → DecisionMapping 생성
└────┬────┘
     ▼
┌─────────┐
│Vectorize│  → 벡터화 여부 결정 및 실행
└────┬────┘
     ▼
┌─────────┐
│  Store  │  → VectorStore + GraphStore 저장
└─────────┘
     │
     ▼
[ProcessedDocument]
```

### 5.2 Stage 인터페이스

```python
class Stage(ABC):
    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """스테이지 처리"""
        pass

    @abstractmethod
    def should_skip(self, context: PipelineContext) -> bool:
        """스킵 조건"""
        pass
```

---

## 6. 외부 의존성

| 패키지 | 용도 | Phase |
|--------|------|-------|
| `pydantic` | 스키마 검증 | 1 |
| `chromadb` | 벡터 저장소 | 2 |
| `openai` | 임베딩 | 2 |
| `tiktoken` | 토큰 카운트 | 2 |
| `httpx` | 비동기 HTTP | 3 |
| `pyyaml` | 온톨로지 로드 | 7 |

---

## 7. 테스트 전략

```
tests/
├── unit/
│   ├── schema/
│   ├── rag/
│   └── ingestion/
├── integration/
│   ├── test_pipeline.py
│   └── test_search.py
└── fixtures/
    ├── sample_documents/
    └── mock_stores.py
```

### 테스트 우선순위
1. `schema/` - 단위 테스트 (직렬화, 검증)
2. `rag/stores/` - 통합 테스트 (실제 ChromaDB)
3. `ingestion/` - 통합 테스트 (E2E 파이프라인)

---

## 8. 마이그레이션 계획

### 8.1 기존 파일 처리

| 현재 파일 | 조치 |
|-----------|------|
| `rag/__init__.py` | 수정 (새 구조 반영) |
| `rag/chunker.py` | 구현 |
| `rag/embedder.py` | 구현 |
| `rag/retriever.py` | 구현 |
| `rag/indexer.py` | 구현 |
| `rag/chroma_store.py` | → `rag/stores/chroma_store.py` 이동 |
| `rag/milvus_store.py` | 삭제 (deprecated) |
| `rag/vector_store.py` | → `rag/stores/base.py` 로 통합 |
| `rag/query_rewriter.py` | → `search/` 로 이동 |

### 8.2 Breaking Changes
- `from agentic_ai_core.rag.chroma_store import ChromaStore`
  → `from agentic_ai_core.rag.stores import ChromaStore`

---

## 9. 체크리스트

### Phase 1: schema/ (예상: 2-3시간)
- [ ] `base.py` - 공통 베이스 클래스
- [ ] `document.py` - Document, Chunk, 관련 Enum
- [ ] `entity.py` - Entity, EntityType
- [ ] `relationship.py` - Relationship, RelationType
- [ ] `decision.py` - DecisionType, DecisionMapping
- [ ] `__init__.py` - 통합 export

### Phase 2: rag/ (예상: 3-4시간)
- [ ] `stores/base.py` - VectorStore ABC
- [ ] `stores/chroma_store.py` - 기존 코드 이동 + 리팩토링
- [ ] `chunker.py` - 청킹 전략
- [ ] `embedder.py` - OpenAI 임베딩
- [ ] `retriever.py` - 검색 로직
- [ ] `indexer.py` - 인덱싱 관리
- [ ] `__init__.py` - 통합 export

### Phase 3: ingestion/ (예상: 4-5시간)
- [ ] `context.py` - PipelineContext
- [ ] `stages/base.py` - Stage ABC
- [ ] 각 스테이지 구현
- [ ] `parsers/` - Notion, Slack 파서
- [ ] `pipeline.py` - 메인 파이프라인

---

## 10. 시작 명령

Phase 1 구현 시작:
```bash
# schema/ 폴더 생성 및 구현
cd packages/agentic-ai-core/src/agentic_ai_core
mkdir -p schema
```


---

 Phase 1 완료

  schema/ 폴더 구현이 완료되었습니다.

  생성된 파일

  schema/
  ├── __init__.py      # 통합 export
  ├── base.py          # SchemaBase, Identifiable, Timestamped, Scorable, Provenanced
  ├── document.py      # Document, Chunk, DocumentType, SourceType, ResolutionLevel
  ├── entity.py        # Entity, EntityType, EntityRef
  ├── relationship.py  # Relationship, RelationType, RelationshipRef
  └── decision.py      # DecisionType, DecisionMapping, DecisionContext

  테스트 결과

  Document created: cb20fceb... type=policy
  Entity created: John Doe type=person
  Relationship created: belongs_to
  Default decision types: 4
  Serialization works: ['created_at', 'updated_at', 'id', ...]

  All schema imports successful!

  핵심 클래스
  ┌─────────────────┬─────────────────────────────────────────────────────────┐
  │     클래스      │                          역할                           │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ Document        │ 문서 전체 스키마 (메타데이터, 청크 참조, 의사결정 매핑) │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ Chunk           │ 벡터화 단위 (임베딩 포함)                               │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ Entity          │ 지식 그래프 노드 (Person, Project, Policy 등)           │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ Relationship    │ 엔티티 간 관계 (belongs_to, manages 등)                 │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ DecisionType    │ 의사결정 유형 정의 (휴가신청, 비용승인 등)              │
  ├─────────────────┼─────────────────────────────────────────────────────────┤
  │ DecisionMapping │ 데이터 → 의사결정 영향도 매핑                           │
  └─────────────────┴─────────────────────────────────────────────────────────┘


● Phase 2 완료

  rag/ 폴더 구현이 완료되었습니다.

  최종 구조

  rag/
  ├── __init__.py           # 통합 export
  ├── stores/
  │   ├── __init__.py
  │   ├── base.py           # VectorStore ABC, SearchResult, IndexStats
  │   └── chroma_store.py   # ChromaDB 구현
  ├── chunker.py            # 4가지 청킹 전략
  ├── embedder.py           # OpenAI, Ollama, SentenceTransformers
  ├── retriever.py          # 검색 + MMR + 다중소스
  └── indexer.py            # 인덱싱 관리 + 증분 인덱서

  테스트 결과

  All RAG imports successful!
  Chunker test: 3 chunks created
    Chunk 0: 47 chars, header=Introduction
    Chunk 1: 33 chars, header=Introduction > Section 1
    Chunk 2: 33 chars, header=Introduction > Section 2

  Phase 2 complete!

  핵심 클래스
  ┌────────────────────┬────────────────────────────────────┐
  │       클래스       │                역할                │
  ├────────────────────┼────────────────────────────────────┤
  │ VectorStore        │ 벡터 저장소 ABC                    │
  ├────────────────────┼────────────────────────────────────┤
  │ ChromaStore        │ ChromaDB 구현 (lazy loading)       │
  ├────────────────────┼────────────────────────────────────┤
  │ SemanticChunker    │ 헤더 기반 의미 단위 청킹           │
  ├────────────────────┼────────────────────────────────────┤
  │ OpenAIEmbedder     │ OpenAI 임베딩 (3-small/large 지원) │
  ├────────────────────┼────────────────────────────────────┤
  │ Retriever          │ 검색 + MMR 다양성                  │
  ├────────────────────┼────────────────────────────────────┤
  │ Indexer            │ 문서 인덱싱 관리                   │
  ├────────────────────┼────────────────────────────────────┤
  │ IncrementalIndexer │ 증분 인덱싱 (변경분만)             │
  └────────────────────┴────────────────────────────────────┘


Phase 3 완료! ingestion/ 모듈이 성공적으로 구현되었습니다.

  생성된 파일 구조

  packages/agentic-ai-core/src/agentic_ai_core/ingestion/
  ├── __init__.py                 # 모듈 exports
  ├── context.py                  # PipelineContext, SourceItem
  ├── pipeline.py                 # Pipeline, PipelineBuilder
  ├── stages/
  │   ├── __init__.py
  │   ├── base.py                 # Stage ABC, ConditionalStage, ParallelStage
  │   ├── parse.py                # ParseStage, DefaultParser
  │   ├── extract.py              # ExtractStage (Entity/Relationship 추출)
  │   ├── infer.py                # InferStage (메타데이터 추론)
  │   ├── score.py                # ScoreStage (Decision Relevance)
  │   ├── vectorize.py            # VectorizeStage (ROI 기반 벡터화)
  │   └── store.py                # StoreStage, InMemoryStores
  └── parsers/
      ├── __init__.py
      ├── base.py                 # Parser ABC, ParserRegistry
      ├── notion_parser.py        # NotionParser, NotionMeetingNoteParser
      └── slack_parser.py         # SlackParser, SlackAnnouncementParser

  핵심 기능
  ┌────────────────┬─────────────────────────────────────────┐
  │    컴포넌트    │                  기능                   │
  ├────────────────┼─────────────────────────────────────────┤
  │ Pipeline       │ 스테이지 체이닝, 배치 처리, 재시도 로직 │
  ├────────────────┼─────────────────────────────────────────┤
  │ ParseStage     │ 문서 타입별 파싱                        │
  ├────────────────┼─────────────────────────────────────────┤
  │ ExtractStage   │ 엔티티/관계 추출 (패턴 매칭)            │
  ├────────────────┼─────────────────────────────────────────┤
  │ InferStage     │ 토픽, 키워드, 언어, 복잡도 추론         │
  ├────────────────┼─────────────────────────────────────────┤
  │ ScoreStage     │ Decision Relevance 계산                 │
  ├────────────────┼─────────────────────────────────────────┤
  │ VectorizeStage │ ROI 기반 벡터화 결정                    │
  ├────────────────┼─────────────────────────────────────────┤
  │ StoreStage     │ Vector/Graph/Document 저장              │
  └────────────────┴─────────────────────────────────────────┘
  사용 예시

  from agentic_ai_core.ingestion import (
      create_default_pipeline,
      SourceItem,
      SourceType,
  )

  # 파이프라인 생성
  pipeline = create_default_pipeline(
      vector_store=chroma_store,
      embedder=openai_embedder
  )

  # 문서 처리
  source = SourceItem(id="doc1", source_type=SourceType.NOTION)
  result = await pipeline.process(source, content="문서 내용...")


● Phase 4 완료! graph/ Knowledge Graph 모듈이 성공적으로 구현되었습니다.

  생성된 파일 구조

  packages/agentic-ai-core/src/agentic_ai_core/graph/
  ├── __init__.py                 # 모듈 exports
  ├── traversal.py                # 그래프 탐색 (BFS, DFS, Dijkstra)
  ├── query.py                    # 그래프 쿼리, 패턴 매칭
  └── stores/
      ├── __init__.py
      ├── base.py                 # GraphStore ABC, PathResult, SubgraphResult
      ├── memory_store.py         # InMemoryGraphStore (개발/테스트용)
      └── neo4j_store.py          # Neo4jGraphStore (프로덕션용)

  핵심 컴포넌트
  ┌────────────────────┬──────────────────────────────────────────────────────┐
  │      컴포넌트      │                         기능                         │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ GraphStore         │ 그래프 저장소 추상 클래스 (Entity/Relationship CRUD) │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ InMemoryGraphStore │ 인메모리 구현 (인덱스 기반 빠른 조회)                │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ Neo4jGraphStore    │ Neo4j 데이터베이스 연동                              │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ GraphTraverser     │ BFS/DFS/Dijkstra 탐색, PageRank 계산                 │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ GraphQuery         │ 조건 기반 쿼리 빌더                                  │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ PatternMatcher     │ Cypher 스타일 패턴 매칭                              │
  ├────────────────────┼──────────────────────────────────────────────────────┤
  │ ContextualSearch   │ 컨텍스트 기반 서브그래프 검색                        │
  └────────────────────┴──────────────────────────────────────────────────────┘
  사용 예시

  from agentic_ai_core.graph import (
      InMemoryGraphStore,
      GraphTraverser,
      GraphQuery,
      TraversalStrategy,
  )
  from agentic_ai_core.schema import Entity, Relationship, EntityType, RelationType

  # 그래프 저장소 생성
  store = InMemoryGraphStore()

  # 엔티티 추가
  await store.add_entity(Entity.create_person("홍길동"))
  await store.add_entity(Entity.create_project("AI Platform"))

  # 관계 추가
  await store.add_relationship(Relationship.create_works_on(person_id, project_id))

  # 이웃 탐색
  neighbors = await store.get_neighbors("entity_id", depth=2)

  # 경로 찾기
  path = await store.find_path("start_id", "end_id", max_depth=5)

  # 쿼리 빌더
  results = await (
      GraphQuery(store)
      .filter_entity_type(EntityType.PERSON)
      .where("name", "like", "홍%")
      .limit(10)
      .execute()
  )

  # PageRank 계산
  traverser = GraphTraverser(store)
  ranks = await traverser.compute_pagerank()


Phase 5 완료! query/ 하이브리드 검색 및 쿼리 최적화 모듈이 성공적으로 구현되었습니다.

  생성된 파일 구조

  packages/agentic-ai-core/src/agentic_ai_core/query/
  ├── __init__.py             # 모듈 exports
  ├── rewriter.py             # 쿼리 재작성, 분석, 확장
  ├── planner.py              # 쿼리 실행 계획 수립
  ├── fusion.py               # 다중 소스 결과 융합
  └── hybrid.py               # 하이브리드 검색기

  핵심 컴포넌트
  ┌──────────────────────┬───────────────────────────────────────────────────────────────┐
  │       컴포넌트       │                             기능                              │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ QueryRewriter        │ 쿼리 분석 (의도, 엔티티, 키워드 추출), 동의어 확장, 쿼리 분해 │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ QueryPlanner         │ 검색 계획 수립, 소스 선택, 실행 순서 결정, 지연시간 예측      │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ ResultFusion         │ RRF, 가중합산, Borda, Cascade 등 다양한 융합 전략             │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ HybridSearcher       │ 벡터 + 그래프 + 키워드 통합 검색                              │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ DiversityReranker    │ MMR 기반 다양성 재정렬                                        │
  ├──────────────────────┼───────────────────────────────────────────────────────────────┤
  │ ContextAwareSearcher │ 대화 컨텍스트 활용 검색                                       │
  └──────────────────────┴───────────────────────────────────────────────────────────────┘
  핵심 기능

  from agentic_ai_core.query import (
      QueryRewriter,
      HybridSearcher,
      HybridSearchConfig,
  )

  # 쿼리 분석
  rewriter = QueryRewriter()
  analysis = rewriter.analyze("휴가 정책이 어떻게 되나요?")
  # → intent=policy_inquiry, is_question=True, keywords=['휴가', '정책']

  # 하이브리드 검색
  searcher = HybridSearcher(
      vector_store=chroma_store,
      graph_store=graph_store,
      embedder=embedder,
      config=HybridSearchConfig(
          vector_weight=0.6,
          graph_weight=0.3,
          fusion_strategy=FusionStrategy.RRF
      )
  )

  result = await searcher.search("프로젝트 담당자 연락처", top_k=10)
  # → results with source_breakdown: {vector: 6, graph: 4}

  융합 전략
  ┌──────────────┬───────────────────────────────────────────┐
  │     전략     │                   설명                    │
  ├──────────────┼───────────────────────────────────────────┤
  │ RRF          │ Reciprocal Rank Fusion - 순위 기반 (기본) │
  ├──────────────┼───────────────────────────────────────────┤
  │ WEIGHTED     │ 소스별 가중치 적용                        │
  ├──────────────┼───────────────────────────────────────────┤
  │ BORDA        │ Borda Count 투표 방식                     │
  ├──────────────┼───────────────────────────────────────────┤
  │ CASCADE      │ 단계적 필터링                             │
  ├──────────────┼───────────────────────────────────────────┤
  │ UNION        │ 합집합 (모든 결과 포함)                   │
  ├──────────────┼───────────────────────────────────────────┤
  │ INTERSECTION │ 교집합 (공통 결과만)                      │
  └──────────────┴───────────────────────────────────────────┘


● Phase 6 완료! scoring/ 스코어링 모듈이 성공적으로 구현되었습니다.

  생성된 파일 구조

  packages/agentic-ai-core/src/agentic_ai_core/scoring/
  ├── __init__.py              # 모듈 exports
  ├── decision_scorer.py       # 의사결정 영향도 계산
  ├── relevance_scorer.py      # 검색 관련성 스코어링
  └── roi_calculator.py        # 벡터화 ROI 계산

  핵심 컴포넌트
  ┌─────────────────┬───────────────────────────────────────────────────────────────┐
  │    컴포넌트     │                             기능                              │
  ├─────────────────┼───────────────────────────────────────────────────────────────┤
  │ DecisionScorer  │ 문서의 의사결정 영향도 계산 (콘텐츠, 엔티티, 메타데이터 기반) │
  ├─────────────────┼───────────────────────────────────────────────────────────────┤
  │ RelevanceScorer │ 검색 관련성 스코어링 (BM25, 의미적 유사도, 정확 매칭)         │
  ├─────────────────┼───────────────────────────────────────────────────────────────┤
  │ ROICalculator   │ 벡터화 ROI 계산 - 가치/비용 기반 벡터화 결정                  │
  └─────────────────┴───────────────────────────────────────────────────────────────┘
  핵심 기능

  from agentic_ai_core.scoring import (
      DecisionScorer,
      RelevanceScorer,
      ROICalculator,
  )

  # 의사결정 영향도 계산
  decision_scorer = DecisionScorer()
  profile = decision_scorer.score_document(
      content="휴가 신청 방법에 대한 정책입니다.",
      doc_type=DocumentType.POLICY,
      entities=[...],
  )
  # → primary_decision="휴가 신청", scores=[4개 타입]

  # 검색 관련성 스코어링
  relevance_scorer = RelevanceScorer()
  score = relevance_scorer.score(
      query="휴가 정책",
      document={"content": "...", "title": "..."},
      semantic_score=0.85
  )
  # → final_score=0.78, signals={semantic=0.85, lexical=0.6, ...}

  # 벡터화 ROI 계산
  roi_calc = ROICalculator()
  roi = roi_calc.calculate(
      content_length=1000,
      relevance_score=0.8,
      decision_influence=0.6
  )
  # → ROI=1141.46, granularity=full, tier=hot

  ROI 기반 벡터화 결정
  ┌───────────┬─────────────┬─────────────┐
  │ ROI 범위  │ 세분화 수준 │ 저장소 계층 │
  ├───────────┼─────────────┼─────────────┤
  │ < 0.5     │ NONE        │ -           │
  ├───────────┼─────────────┼─────────────┤
  │ 0.5 ~ 1.0 │ METADATA    │ COLD        │
  ├───────────┼─────────────┼─────────────┤
  │ 1.0 ~ 2.0 │ SUMMARY     │ WARM        │
  ├───────────┼─────────────┼─────────────┤
  │ 2.0 ~ 5.0 │ CHUNKS      │ WARM        │
  ├───────────┼─────────────┼─────────────┤
  │ > 5.0     │ FULL        │ HOT         │
  └───────────┴─────────────┴─────────────┘

  ● Phase 7 완료! ontology/ 모듈 구현 완료:

  ontology/
  ├── __init__.py        # 모듈 exports
  ├── loader.py          # YAML/JSON 온톨로지 로더
  └── validator.py       # 스키마 검증기

  구현 내용:
  - OntologyLoader: YAML/JSON 파일에서 온톨로지 로드
  - Ontology: 엔티티/관계/의사결정 타입 정의 구조
  - OntologyValidator: 스키마 무결성 검증 (순환상속, 참조무결성 등)
  - EntityValidator: 런타임 엔티티 데이터 검증
  - load_default_ontology(): 내장 internal_ops 온톨로지

  테스트 결과:
  - 기본 온톨로지 로드 성공 (internal_ops v1.0.0)
  - 엔티티 타입 6개, 관계 타입 6개, 의사결정 타입 4개
  - 검증 통과 (errors=0, warnings=1)

  전체 모듈 현황 (Phase 1-7 완료):
  - api, schema, rag, ingestion, graph, query, scoring, ontology

● Phase 8 완료! lifecycle/ 모듈 구현 완료:

  lifecycle/
  ├── __init__.py        # 모듈 exports
  ├── manager.py         # 생명주기/해상도 관리
  └── scheduler.py       # 정리 작업 스케줄러

  구현 내용:
  ┌──────────────────────┬──────────────────────────────────────────────────────────┐
  │       컴포넌트       │                           기능                           │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ LifecycleManager     │ 문서 생명주기 관리 (DRAFT→ACTIVE→STALE→ARCHIVED→DELETED) │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ ResolutionLevel      │ 해상도 수준 (MINIMAL, STANDARD, DETAILED, FULL)          │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ StorageTier          │ 저장소 계층 (HOT, WARM, COLD, GLACIER)                   │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ AccessPattern        │ 접근 패턴 추적 (빈도, 최근 접근 등)                      │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ LifecycleScheduler   │ 유지보수 작업 스케줄러                                   │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ PolicyEvaluationTask │ 정책 평가 작업                                           │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ CleanupTask          │ 삭제 문서 정리                                           │
  ├──────────────────────┼──────────────────────────────────────────────────────────┤
  │ TierMigrationTask    │ 저장소 계층 마이그레이션                                 │
  └──────────────────────┴──────────────────────────────────────────────────────────┘
  테스트 결과:
  - 문서 등록, 접근 기록, 상태 전이 정상 동작
  - 해상도 업그레이드/다운그레이드 정상
  - 통계 수집 정상

  ---
  전체 agentic-ai-core 완성 (Phase 1-8):
  ┌───────┬───────────┬─────────┐
  │ Phase │   모듈    │ Exports │
  ├───────┼───────────┼─────────┤
  │ 1     │ schema    │ 33      │
  ├───────┼───────────┼─────────┤
  │ 2     │ rag       │ 37      │
  ├───────┼───────────┼─────────┤
  │ 3     │ ingestion │ 41      │
  ├───────┼───────────┼─────────┤
  │ 4     │ graph     │ 20      │
  ├───────┼───────────┼─────────┤
  │ 5     │ query     │ 23      │
  ├───────┼───────────┼─────────┤
  │ 6     │ scoring   │ 18      │
  ├───────┼───────────┼─────────┤
  │ 7     │ ontology  │ 15      │
  ├───────┼───────────┼─────────┤
  │ 8     │ lifecycle │ 24      │
  ├───────┼───────────┼─────────┤
  │ -     │ api       │ 4       │
  └───────┴───────────┴─────────┘
  총 9개 모듈, 215개 exports - 패키지 완성!


---


  ┌─────────────────────────────────────────────────────────┐
  │  1. schema/        ← 모든 것의 기초 (여기서 시작!)       │
  │     └─ 데이터 구조 정의                                 │
  ├─────────────────────────────────────────────────────────┤
  │  2. ontology/      ← 도메인 모델링 (독립적)             │
  │     └─ 엔티티/관계 타입 정의                            │
  ├─────────────────────────────────────────────────────────┤
  │  3. rag/           ← 벡터 검색 기본                     │
  │     └─ 청킹 → 임베딩 → 저장 → 검색                     │
  ├─────────────────────────────────────────────────────────┤
  │  4. graph/         ← 지식 그래프                        │
  │     └─ 엔티티/관계 저장 및 탐색                         │
  ├─────────────────────────────────────────────────────────┤
  │  5. scoring/       ← 스코어링 시스템                    │
  │     └─ 의사결정 영향도, 관련성, ROI                     │
  ├─────────────────────────────────────────────────────────┤
  │  6. ingestion/     ← 데이터 수집 파이프라인             │
  │     └─ 위 모듈들을 조합한 처리 흐름                     │
  ├─────────────────────────────────────────────────────────┤
  │  7. query/         ← 하이브리드 검색                    │
  │     └─ rag + graph 통합 검색                           │
  ├─────────────────────────────────────────────────────────┤
  │  8. lifecycle/     ← 생명주기 관리                      │
  │     └─ 전체 시스템 운영                                 │
  └─────────────────────────────────────────────────────────┘

  ---
  모듈별 학습 가이드

  1. schema/ (1-2시간)

  핵심 개념: 데이터 모델링, Pydantic, 직렬화
  ┌──────┬─────────────────┬───────────────────────────────┐
  │ 순서 │      파일       │          핵심 클래스          │
  ├──────┼─────────────────┼───────────────────────────────┤
  │ 1    │ base.py         │ SchemaBase, Identifiable      │
  ├──────┼─────────────────┼───────────────────────────────┤
  │ 2    │ document.py     │ Document, Chunk, DocumentType │
  ├──────┼─────────────────┼───────────────────────────────┤
  │ 3    │ entity.py       │ Entity, EntityType            │
  ├──────┼─────────────────┼───────────────────────────────┤
  │ 4    │ relationship.py │ Relationship, RelationType    │
  ├──────┼─────────────────┼───────────────────────────────┤
  │ 5    │ decision.py     │ DecisionType, DecisionMapping │
  └──────┴─────────────────┴───────────────────────────────┘
  포인트: 이 구조들이 나머지 모든 모듈에서 사용됨

  ---
  2. ontology/ (30분)

  핵심 개념: 도메인 온톨로지, YAML 스키마, 검증
  ┌──────┬──────────────┬──────────────────────────┐
  │ 순서 │     파일     │       핵심 클래스        │
  ├──────┼──────────────┼──────────────────────────┤
  │ 1    │ loader.py    │ OntologyLoader, Ontology │
  ├──────┼──────────────┼──────────────────────────┤
  │ 2    │ validator.py │ OntologyValidator        │
  └──────┴──────────────┴──────────────────────────┘
  포인트: DEFAULT_ONTOLOGY_YAML 예시로 구조 파악

  ---
  3. rag/ (2-3시간) ⭐ 중요

  핵심 개념: 청킹, 임베딩, 벡터 검색, MMR
  ┌──────┬────────────────────────┬───────────────────────────────────┐
  │ 순서 │          파일          │            핵심 클래스            │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 1    │ stores/base.py         │ VectorStore (ABC)                 │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 2    │ stores/chroma_store.py │ ChromaStore                       │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 3    │ chunker.py             │ SemanticChunker, ChunkingStrategy │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 4    │ embedder.py            │ OpenAIEmbedder                    │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 5    │ retriever.py           │ Retriever (MMR 다양성)            │
  ├──────┼────────────────────────┼───────────────────────────────────┤
  │ 6    │ indexer.py             │ Indexer, IncrementalIndexer       │
  └──────┴────────────────────────┴───────────────────────────────────┘
  포인트: RAG 파이프라인의 핵심 - 청킹→임베딩→검색 흐름 이해

  ---
  4. graph/ (1-2시간)

  핵심 개념: 그래프 DB, 탐색 알고리즘, 패턴 매칭
  ┌──────┬────────────────────────┬─────────────────────────────────────┐
  │ 순서 │          파일          │             핵심 클래스             │
  ├──────┼────────────────────────┼─────────────────────────────────────┤
  │ 1    │ stores/base.py         │ GraphStore (ABC)                    │
  ├──────┼────────────────────────┼─────────────────────────────────────┤
  │ 2    │ stores/memory_store.py │ InMemoryGraphStore                  │
  ├──────┼────────────────────────┼─────────────────────────────────────┤
  │ 3    │ traversal.py           │ GraphTraverser (BFS, DFS, PageRank) │
  ├──────┼────────────────────────┼─────────────────────────────────────┤
  │ 4    │ query.py               │ GraphQuery, PatternMatcher          │
  └──────┴────────────────────────┴─────────────────────────────────────┘
  포인트: 엔티티 간 관계를 활용한 컨텍스트 확장

  ---
  5. scoring/ (1시간)

  핵심 개념: 스코어링 알고리즘, ROI 계산
  ┌──────┬─────────────────────┬────────────────────────────────┐
  │ 순서 │        파일         │          핵심 클래스           │
  ├──────┼─────────────────────┼────────────────────────────────┤
  │ 1    │ decision_scorer.py  │ DecisionScorer                 │
  ├──────┼─────────────────────┼────────────────────────────────┤
  │ 2    │ relevance_scorer.py │ RelevanceScorer (BM25, 의미적) │
  ├──────┼─────────────────────┼────────────────────────────────┤
  │ 3    │ roi_calculator.py   │ ROICalculator (벡터화 결정)    │
  └──────┴─────────────────────┴────────────────────────────────┘
  포인트: ROI = (Expected_Value × Query_Probability) / Total_Cost

  ---
  6. ingestion/ (2-3시간) ⭐ 중요

  핵심 개념: 파이프라인 패턴, Stage 체이닝
  ┌──────┬─────────────────────┬─────────────────────────────┐
  │ 순서 │        파일         │         핵심 클래스         │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 1    │ context.py          │ PipelineContext, SourceItem │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 2    │ stages/base.py      │ Stage (ABC)                 │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 3    │ stages/parse.py     │ ParseStage                  │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 4    │ stages/extract.py   │ ExtractStage                │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 5    │ stages/score.py     │ ScoreStage                  │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 6    │ stages/vectorize.py │ VectorizeStage              │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 7    │ pipeline.py         │ Pipeline, PipelineBuilder   │
  ├──────┼─────────────────────┼─────────────────────────────┤
  │ 8    │ parsers/            │ NotionParser, SlackParser   │
  └──────┴─────────────────────┴─────────────────────────────┘
  포인트: 전체 데이터 처리 흐름 - Fetch→Parse→Extract→Score→Vectorize→Store

  ---
  7. query/ (1-2시간)

  핵심 개념: 하이브리드 검색, 결과 융합
  ┌──────┬─────────────┬───────────────────────────┐
  │ 순서 │    파일     │        핵심 클래스        │
  ├──────┼─────────────┼───────────────────────────┤
  │ 1    │ rewriter.py │ QueryRewriter (의도 분석) │
  ├──────┼─────────────┼───────────────────────────┤
  │ 2    │ fusion.py   │ ResultFusion (RRF, Borda) │
  ├──────┼─────────────┼───────────────────────────┤
  │ 3    │ planner.py  │ QueryPlanner              │
  ├──────┼─────────────┼───────────────────────────┤
  │ 4    │ hybrid.py   │ HybridSearcher            │
  └──────┴─────────────┴───────────────────────────┘
  포인트: 벡터 + 그래프 + 키워드 검색 통합

  ---
  8. lifecycle/ (30분)

  핵심 개념: 문서 상태 관리, 스케줄링
  ┌──────┬──────────────┬───────────────────────────────────┐
  │ 순서 │     파일     │            핵심 클래스            │
  ├──────┼──────────────┼───────────────────────────────────┤
  │ 1    │ manager.py   │ LifecycleManager, ResolutionLevel │
  ├──────┼──────────────┼───────────────────────────────────┤
  │ 2    │ scheduler.py │ LifecycleScheduler                │
  └──────┴──────────────┴───────────────────────────────────┘
  포인트: 운영 관점의 데이터 관리

  ---
  실습 추천

  # 1. 각 모듈 import 해보기
  from agentic_ai_core.schema import Document, Entity
  from agentic_ai_core.rag import SemanticChunker, ChromaStore

  # 2. 간단한 객체 생성해보기
  doc = Document(title="테스트", content="내용")
  entity = Entity.create_person("홍길동")

  # 3. 청킹/검색 흐름 따라가보기
  chunker = SemanticChunker()
  chunks = chunker.chunk("# 제목\n내용...")

  어떤 모듈부터 시작할까요? 제가 해당 모듈의 코드를 설명해드릴 수 있어요.