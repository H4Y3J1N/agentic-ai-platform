# Module Guide

패키지 선택 및 사용 가이드

## 패키지 구조

```
packages/
├── core/        # 필수 - 모든 도메인에서 사용
├── knowledge/   # 선택 - 지식그래프 필요 시
└── decision/    # 선택 - 의사결정 지원 필요 시
```

## 어떤 패키지를 사용해야 하나?

| 도메인 유형 | 사용 패키지 |
|------------|------------|
| 단순 Q&A 챗봇 | `core` |
| 문서 검색 RAG | `core` |
| 지식 관리 시스템 | `core` + `knowledge` |
| 내부 운영 도구 | `core` + `knowledge` + `decision` |
| 이커머스 어시스턴트 | `core` (+ 도메인별 스키마) |

---

## core (agentic_ai_core)

**모든 도메인에서 공통으로 사용하는 기능**

경로: `packages/core/agentic_ai_core/`

| 모듈 | 용도 |
|------|------|
| `llm/` | LLM 게이트웨이 (OpenAI, Anthropic, Ollama) |
| `rag/` | 벡터 검색 (Chunker, Embedder, Retriever, ChromaStore) |
| `ingestion/` | 데이터 파이프라인 (Pipeline, Stages) |
| `query/` | 하이브리드 검색, 쿼리 리라이팅 |
| `scoring/` | 관련성 스코어링, ROI 계산 |
| `api/` | SSE, WebSocket |
| `orchestrator/` | 에이전트 패턴 |

```python
from agentic_ai_core.rag import ChromaStore, OpenAIEmbedder, Retriever
from agentic_ai_core.ingestion import Pipeline, PipelineBuilder
from agentic_ai_core.query import HybridSearcher
```

---

## knowledge (agentic_ai_knowledge)

**지식 그래프가 필요한 도메인용 확장**

경로: `packages/knowledge/agentic_ai_knowledge/`

| 모듈 | 용도 |
|------|------|
| `schema/` | Entity, Relationship, EntityType, RelationType |
| `graph/` | GraphStore, InMemoryGraphStore, Neo4jGraphStore |
| `ontology/` | 온톨로지 로더, 검증기 |
| `extraction/` | 엔티티/관계 추출 스테이지 |

**사용 시점:**
- 문서에서 엔티티(사람, 팀, 프로젝트)를 추출해야 할 때
- 엔티티 간 관계를 그래프로 저장/조회해야 할 때

```python
from agentic_ai_knowledge import (
    Entity, EntityType, Relationship,
    InMemoryGraphStore, ExtractStage,
)
```

---

## decision (agentic_ai_decision)

**의사결정 지원이 필요한 도메인용 확장**

경로: `packages/decision/agentic_ai_decision/`

| 모듈 | 용도 |
|------|------|
| `schema/` | DecisionType, DecisionMapping |
| `scoring/` | DecisionScorer |
| `lifecycle/` | LifecycleManager, LifecycleScheduler |

**사용 시점:**
- 데이터가 의사결정에 미치는 영향도를 계산해야 할 때
- 데이터 생명주기 관리가 필요할 때

```python
from agentic_ai_decision import (
    DecisionType, DecisionScorer, LifecycleManager,
)
```

---

## 조합 패턴

### 패턴 1: 단순 RAG

```python
from agentic_ai_core.rag import ChromaStore, OpenAIEmbedder
from agentic_ai_core.ingestion import create_lightweight_pipeline

pipeline = create_lightweight_pipeline(
    vector_store=ChromaStore(),
    embedder=OpenAIEmbedder()
)
```

### 패턴 2: 지식 그래프 RAG

```python
from agentic_ai_core.rag import ChromaStore
from agentic_ai_core.query import HybridSearcher
from agentic_ai_knowledge import InMemoryGraphStore, ExtractStage

searcher = HybridSearcher(
    vector_store=ChromaStore(),
    graph_store=InMemoryGraphStore()
)
```

### 패턴 3: 내부 운영 도구

```python
from agentic_ai_core.rag import ChromaStore
from agentic_ai_knowledge import InMemoryGraphStore, ExtractStage
from agentic_ai_decision import DecisionScorer, LifecycleManager

# 전체 파이프라인
pipeline = PipelineBuilder() \
    .add(ParseStage) \
    .add(ExtractStage) \
    .add(VectorizeStage) \
    .build()
```

---

## 의존성

```
core (필수)
  ↑
knowledge (선택)
  ↑
decision (선택, knowledge 선택적 의존)
```
