# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular Agentic AI platform designed as a reusable template. Core philosophy: "AI engine stays the same, only swap out domain knowledge." Clone the platform, configure domain-specific settings in YAML, and deploy.

**Tech Stack**: Python 3.11+, FastAPI, Poetry, PostgreSQL, Redis, ChromaDB/Milvus, Langfuse, Docker, vLLM

## Project Structure

```
agentic-ai-platform/
├── packages/                    # 재사용 가능한 코어 라이브러리 (레이어별 분리)
│   ├── core/                    # 기본 인프라 (LLM, RAG, API)
│   ├── pipeline/                # 데이터 처리 (ingestion, query, scoring, evaluation)
│   ├── agents/                  # 에이전트 (orchestrator, base agents, tools)
│   ├── knowledge/               # 지식그래프 (entity, graph, ontology)
│   ├── decision/                # 의사결정 (scoring, lifecycle)
│   └── serving/                 # 로컬 모델 서빙 (vLLM, LoRA, quantization)
├── services/                    # 도메인별 서비스
│   ├── sample/                  # 템플릿 (새 서비스 생성 시 복사)
│   └── internal-ops/            # 내부 운영 도구 서비스
└── docs/                        # 문서
```

## Package Structure

| 도메인 유형 | 사용 패키지 |
|------------|------------|
| 단순 Q&A 챗봇 | `core` + `agents` |
| 문서 검색 RAG | `core` + `pipeline` + `agents` |
| 지식 관리 시스템 | `core` + `pipeline` + `agents` + `knowledge` |
| 내부 운영 도구 | `core` + `pipeline` + `agents` + `knowledge` + `decision` |
| 로컬 LLM 서빙 | `core` + `serving` |
| 멀티테넌트 LoRA | `core` + `agents` + `serving` |

자세한 사용법은 `docs/MODULE_GUIDE.md` 참조

## Common Commands

```bash
# Start service locally (from services/internal-ops)
uvicorn internal_ops_service.api.main:app --reload --port 8000

# Docker development
cd services/internal-ops/docker && docker-compose up -d

# Run tests
pytest tests/
pytest tests/unit/test_rate_limiter.py

# Index Notion workspace (internal-ops only)
python scripts/index_notion.py full

# Start vLLM server (serving package)
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-lora
```

## Architecture

### packages/core (기본 인프라)

```
packages/core/agentic_core/
├── llm/             # LLM gateway (OpenAI, Anthropic)
│   └── providers/   # Provider implementations
├── rag/             # Vector search
│   └── stores/      # ChromaStore, MilvusStore
├── api/             # SSE, WebSocket, FastAPI utilities
├── schema/          # 기본 스키마 (Document, Chunk)
├── security/        # RBAC, JWT, rate limiting
├── database/        # DB connection utilities
└── observability/   # Langfuse integration
```

### packages/pipeline (데이터 처리)

```
packages/pipeline/agentic_pipeline/
├── ingestion/       # 데이터 수집 파이프라인
│   ├── stages/      # Parse, Infer, Vectorize, Store
│   └── parsers/     # Notion, Slack 등
├── query/           # 쿼리 처리
│   ├── rewriter.py  # 쿼리 리라이팅
│   ├── planner.py   # 쿼리 계획
│   ├── hybrid.py    # 하이브리드 검색
│   └── fusion.py    # 결과 융합
├── scoring/         # 스코어링
│   ├── relevance_scorer.py
│   └── roi_calculator.py
└── evaluation/      # 품질 평가
    ├── rag_evaluator.py
    ├── llm_evaluator.py
    └── benchmark_runner.py
```

### packages/agents (에이전트)

```
packages/agents/agentic_agents/
├── orchestrator/    # 오케스트레이션
│   ├── patterns/    # supervisor, hierarchy, collaborative, sequential
│   ├── message_bus.py
│   └── workflow_engine.py
├── base/            # 기본 에이전트
│   ├── base_agent.py
│   ├── streaming_agent.py
│   └── agent_registry.py
└── tools/           # 기본 도구
    ├── base_tool.py
    └── tool_registry.py
```

### packages/knowledge (지식그래프)

```
packages/knowledge/agentic_knowledge/
├── schema/          # Entity, Relationship
├── graph/           # GraphStore (memory, neo4j)
│   └── stores/
├── ontology/        # OntologyLoader, Validator
└── extraction/      # 엔티티/관계 추출
```

### packages/decision (의사결정)

```
packages/decision/agentic_decision/
├── schema/          # DecisionType, DecisionMapping
├── scoring/         # DecisionScorer
└── lifecycle/       # LifecycleManager, Scheduler
```

### packages/serving (로컬 모델 서빙)

```
packages/serving/agentic_serving/
├── vllm/            # vLLM 서버 연동
│   ├── config.py    # VLLMConfig, VLLMModelConfig
│   └── provider.py  # VLLMProvider
├── lora/            # LoRA 어댑터 관리
│   ├── config.py    # LoRAAdapter, TenantLoRAMapping
│   ├── registry.py  # LoRAAdapterRegistry
│   └── loader.py    # LoRALoader
└── quantization/    # 양자화 설정
    ├── config.py    # QLoRA, EXL2, GPTQ, AWQ
    └── utils.py     # 메모리 추정, 추천
```

### services (도메인 서비스)

```
services/{service-name}/
├── {service_name}_service/      # 모든 Python 코드
│   ├── agents/                  # 도메인 에이전트
│   ├── tools/                   # 도메인 도구
│   ├── integrations/            # 외부 서비스 연동 (선택)
│   └── api/                     # FastAPI 앱
│       ├── routes/
│       ├── middleware/
│       ├── schemas/
│       └── main.py
├── config/                      # YAML 설정
├── docker/                      # Dockerfile, docker-compose.yml
├── nginx/                       # Nginx 설정
├── scripts/                     # CLI 스크립트 (선택)
├── tests/
└── pyproject.toml
```

## Key Patterns

**API Communication**:
- REST: `POST /agent/chat`
- SSE: `GET /agent/chat/stream`
- WebSocket: `WS /ws/chat/{session_id}`

**Orchestration** (`agents/orchestrator/patterns/`):
- `supervisor.py`: Delegates to specialist agents
- `hierarchy.py`: Multi-level delegation
- `collaborative.py`: Parallel execution, merge results
- `sequential.py`: Pipeline chaining

**Ingestion Pipeline**: `Parse → Infer → Vectorize → Store`
(+ `Extract`, `Score` stages when using knowledge/decision packages)

## Configuration

YAML configs in `services/{service}/config/`:
- `domain.yaml` - Service identity, agent list
- `rag.yaml` - Embedding model, chunk size, retrieval settings
- `llm.yaml` - LLM provider settings
- `api.yaml` - CORS, timeouts
- `rate_limit.yaml` - Token bucket settings
- `serving.yaml` - (선택) vLLM, LoRA, 양자화 설정

Environment variables:
```bash
OPENAI_API_KEY, ANTHROPIC_API_KEY          # LLM providers
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY   # Observability
NOTION_API_KEY, NOTION_WORKSPACE_ID        # Integrations (internal-ops)
DATABASE_URL, REDIS_URL, CHROMA_PERSIST_DIR
MILVUS_URI                                 # Milvus connection (optional)
VLLM_BASE_URL                              # vLLM server URL (serving package)
LORA_ADAPTERS_PATH                         # LoRA adapters directory
```

## Creating a New Domain Service

1. Copy the sample service:
```bash
cp -r services/sample services/{new-service-name}
cd services/{new-service-name}
mv sample_service {new_service_name}_service
```

2. Update references (sample → {new_service_name}):
   - `pyproject.toml`
   - `docker/Dockerfile`
   - `docker/docker-compose.yml`
   - `nginx/sample.conf` → `nginx/{new_service_name}.conf`
   - `config/domain.yaml`

3. Choose packages in `pyproject.toml`:
```toml
agentic-core = {path = "../../packages/core", develop = true}
agentic-pipeline = {path = "../../packages/pipeline", develop = true}
agentic-agents = {path = "../../packages/agents", develop = true}
# agentic-knowledge = {path = "../../packages/knowledge", develop = true}
# agentic-decision = {path = "../../packages/decision", develop = true}
# agentic-serving = {path = "../../packages/serving", develop = true}
```

4. Implement domain logic:
   - `{name}_service/agents/` - Domain agents
   - `{name}_service/tools/` - Domain tools

See `services/sample/README.md` for detailed guide.

## Usage Examples

### Vector Store (Milvus Lite)

```python
from agentic_core.rag.stores import MilvusStore

# 로컬 파일 기반 (Milvus Lite)
store = MilvusStore(
    collection_name="documents",
    uri="./milvus.db",
    dimension=1536,
    metric_type="COSINE"
)

await store.insert(ids, texts, embeddings, metadatas)
results = await store.search(query_embedding, top_k=5)
```

### Quality Evaluation

```python
from agentic_pipeline.evaluation import RAGEvaluator, EvaluationConfig, MetricType

config = EvaluationConfig(
    metrics=[MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCE],
    thresholds={MetricType.FAITHFULNESS: 0.7}
)
evaluator = RAGEvaluator(config=config, llm_judge=my_llm_call)

results = await evaluator.evaluate(
    query="질문",
    response="응답",
    context=["컨텍스트1", "컨텍스트2"],
    ground_truth="정답"
)
```

### vLLM Provider

```python
from agentic_serving import VLLMProvider, VLLMConfig

config = VLLMConfig(
    base_url="http://localhost:8000/v1",
    default_model="meta-llama/Llama-2-7b-hf",
    enable_lora_routing=True
)

async with VLLMProvider(config) as provider:
    response = await provider.chat(messages)

    # LoRA 어댑터 사용
    response = await provider.chat(messages, lora_adapter="customer-service-adapter")
```

### Multi-tenant LoRA

```python
from agentic_serving import LoRAAdapterRegistry, LoRALoader, LoRAConfig, LoRAAdapter

config = LoRAConfig(
    adapters_base_path="./lora_adapters",
    max_loaded_adapters=4,
    enable_tenant_routing=True
)

registry = LoRAAdapterRegistry(config)
registry.register(LoRAAdapter(
    name="tenant-a-adapter",
    path="./lora_adapters/tenant_a",
    base_model="meta-llama/Llama-2-7b-hf"
))

loader = LoRALoader(config, registry)
adapter_name = await loader.get_for_tenant("tenant-a")
```

## Health Endpoints

- `/health` - Liveness probe
- `/readiness` - Readiness probe
- `/startup` - Startup probe

## Package Dependencies

```
core ─────────────────────────────────────────┐
  ↑                                           │
pipeline ──────────────────────────┐          │
  ↑                                │          │
agents ────────────────┐           │          │
  ↑                    │           │          │
knowledge ────┐        │           │          │
  ↑           │        │           │          │
decision      │        │           │          │
              ↓        ↓           ↓          ↓
           serving (optional, depends only on core)
```

## Evaluation Metrics

### RAG Metrics
| Metric | Description | Good Score |
|--------|-------------|------------|
| Faithfulness | 응답이 컨텍스트에 충실한지 | ≥ 0.7 |
| Answer Relevance | 응답이 질문에 관련있는지 | ≥ 0.7 |
| Context Relevance | 검색된 컨텍스트가 질문에 관련있는지 | ≥ 0.6 |
| Context Recall | 정답에 필요한 정보가 컨텍스트에 있는지 | ≥ 0.6 |

### LLM Metrics
| Metric | Description | Good Score |
|--------|-------------|------------|
| Coherence | 논리적 일관성 | ≥ 0.7 |
| Fluency | 문법적 유창성 | ≥ 0.7 |
| Toxicity | 유해성 (낮을수록 좋음) | ≤ 0.1 |
| Hallucination | 환각 (낮을수록 좋음) | ≤ 0.2 |

## Quantization Methods

| Method | Use Case | Speed | Quality | VRAM Saving |
|--------|----------|-------|---------|-------------|
| QLoRA | Fine-tuning | Medium | High | ~75% |
| EXL2 | Inference | Fast | High | ~75% |
| GPTQ | Inference | Fast | Medium | ~75% |
| AWQ | Inference | Fast | High | ~75% |
| FP8 | Inference (vLLM) | Fast | Very High | ~50% |
