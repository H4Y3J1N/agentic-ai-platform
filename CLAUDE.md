# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular Agentic AI platform designed as a reusable template. Core philosophy: "AI engine stays the same, only swap out domain knowledge." Clone the platform, configure domain-specific settings in YAML, and deploy.

**Tech Stack**: Python 3.11+, FastAPI, Poetry, PostgreSQL, Redis, ChromaDB/Milvus, Langfuse, Docker, vLLM

## Project Structure

```
agentic-ai-platform/
├── packages/                    # 재사용 가능한 코어 라이브러리
│   ├── core/                    # 필수 - 모든 도메인에서 사용
│   ├── knowledge/               # 선택 - 지식그래프 필요 시
│   ├── decision/                # 선택 - 의사결정 지원 필요 시
│   └── serving/                 # 선택 - 로컬 모델 서빙 시 (vLLM, LoRA, 양자화)
├── services/                    # 도메인별 서비스
│   ├── sample/                  # 템플릿 (새 서비스 생성 시 복사)
│   └── internal-ops/            # 내부 운영 도구 서비스
└── docs/                        # 문서
```

## Package Structure

| 도메인 유형 | 사용 패키지 |
|------------|------------|
| 단순 Q&A 챗봇 | `core` |
| 문서 검색 RAG | `core` |
| 지식 관리 시스템 | `core` + `knowledge` |
| 내부 운영 도구 | `core` + `knowledge` + `decision` |
| 로컬 LLM 서빙 | `core` + `serving` |
| 멀티테넌트 LoRA | `core` + `serving` |

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

### packages/core (공통)

```
packages/core/agentic_ai_core/
├── llm/             # LLM gateway (OpenAI, Anthropic, Ollama)
├── rag/             # Vector search: chunker, embedder, retriever
│   └── stores/      # ChromaStore, MilvusStore
├── ingestion/       # Data pipeline: Pipeline, Stages
├── query/           # Hybrid search, query rewriting, fusion
├── scoring/         # Relevance scoring, ROI calculation
├── evaluation/      # 품질 평가: RAGEvaluator, LLMEvaluator, BenchmarkRunner
├── orchestrator/    # Agent patterns: supervisor, hierarchy, collaborative, sequential
├── api/             # SSE, WebSocket
├── agents/          # Base agent classes
├── tools/           # Base tool classes
├── security/        # RBAC, JWT, rate limiting
└── schema/          # Document, Chunk (기본 스키마만)
```

### packages/knowledge (지식그래프용)

```
packages/knowledge/agentic_ai_knowledge/
├── schema/          # Entity, Relationship, EntityType, RelationType
├── graph/           # GraphStore, InMemoryGraphStore, Neo4jGraphStore
├── ontology/        # OntologyLoader, OntologyValidator
└── extraction/      # ExtractStage (엔티티/관계 추출)
```

### packages/decision (의사결정 지원용)

```
packages/decision/agentic_ai_decision/
├── schema/          # DecisionType, DecisionMapping
├── scoring/         # DecisionScorer
└── lifecycle/       # LifecycleManager, LifecycleScheduler
```

### packages/serving (로컬 모델 서빙용)

```
packages/serving/agentic_ai_serving/
├── vllm/            # vLLM 서버 연동
│   ├── config.py    # VLLMConfig, VLLMModelConfig
│   └── provider.py  # VLLMProvider (OpenAI-compatible API)
├── lora/            # LoRA 어댑터 관리
│   ├── config.py    # LoRAAdapter, TenantLoRAMapping, LoRAConfig
│   ├── registry.py  # LoRAAdapterRegistry (등록/조회/테넌트 매핑)
│   └── loader.py    # LoRALoader (LRU 캐시, 동적 로딩/언로딩)
└── quantization/    # 양자화 설정
    ├── config.py    # QLoRAConfig, EXL2Config, GPTQConfig, AWQConfig
    └── utils.py     # 메모리 추정, 추천 양자화, 설정 검증
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

**Orchestration** (`orchestrator/patterns/`):
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
agentic-ai-core = {path = "../../packages/core", develop = true}
# agentic-ai-knowledge = {path = "../../packages/knowledge", develop = true}
# agentic-ai-decision = {path = "../../packages/decision", develop = true}
# agentic-ai-serving = {path = "../../packages/serving", develop = true}
```

4. Implement domain logic:
   - `{name}_service/agents/` - Domain agents
   - `{name}_service/tools/` - Domain tools

See `services/sample/README.md` for detailed guide.

## Usage Examples

### Vector Store (Milvus Lite)

```python
from agentic_ai_core.rag.stores import MilvusStore

# 로컬 파일 기반 (Milvus Lite)
store = MilvusStore(
    collection_name="documents",
    uri="./milvus.db",
    dimension=1536,
    metric_type="COSINE"
)

# 서버 연결
store = MilvusStore(
    collection_name="documents",
    uri="http://localhost:19530"
)

await store.insert(ids, texts, embeddings, metadatas)
results = await store.search(query_embedding, top_k=5)
```

### Quality Evaluation

```python
from agentic_ai_core.evaluation import RAGEvaluator, LLMEvaluator, EvaluationConfig

# RAG 평가
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

# 배치 평가
batch_results = await evaluator.evaluate_batch(samples)
aggregates = evaluator.get_aggregate_scores(batch_results)
```

### vLLM Provider

```python
from agentic_ai_serving import VLLMProvider, VLLMConfig

config = VLLMConfig(
    base_url="http://localhost:8000/v1",
    default_model="meta-llama/Llama-2-7b-hf",
    enable_lora_routing=True
)

async with VLLMProvider(config) as provider:
    # 일반 생성
    response = await provider.chat(messages)

    # 스트리밍
    async for chunk in provider.chat_stream(messages):
        print(chunk.content, end="")

    # LoRA 어댑터 사용
    response = await provider.chat(messages, lora_adapter="customer-service-adapter")
```

### Multi-tenant LoRA

```python
from agentic_ai_serving import LoRAAdapterRegistry, LoRALoader, LoRAConfig, LoRAAdapter

# 레지스트리 설정
config = LoRAConfig(
    adapters_base_path="./lora_adapters",
    max_loaded_adapters=4,
    enable_tenant_routing=True
)

registry = LoRAAdapterRegistry(config)

# 어댑터 등록
registry.register(LoRAAdapter(
    name="tenant-a-adapter",
    path="./lora_adapters/tenant_a",
    base_model="meta-llama/Llama-2-7b-hf",
    rank=16,
    alpha=32
))

# 테넌트 매핑
registry.set_tenant_mapping(TenantLoRAMapping(
    tenant_id="tenant-a",
    adapter_name="tenant-a-adapter"
))

# 로더로 동적 로딩
loader = LoRALoader(config, registry)
adapter_name = await loader.get_for_tenant("tenant-a")
```

### Quantization

```python
from agentic_ai_serving.quantization import (
    QuantizationConfig, QuantizationMethod, QLoRAConfig,
    estimate_memory_usage, get_recommended_quantization
)

# 메모리 추정
estimate = estimate_memory_usage(
    model_params_billions=7.0,
    quantization_method=QuantizationMethod.QLORA,
    context_length=4096
)
print(f"예상 VRAM: {estimate.total_memory_gb}GB")

# 추천 양자화
method, bits, config = get_recommended_quantization(
    model_params_billions=13.0,
    available_vram_gb=24.0,
    use_case="inference",
    priority="speed"
)

# QLoRA 설정
quant_config = QuantizationConfig(
    method=QuantizationMethod.QLORA,
    qlora=QLoRAConfig(
        bits=4,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
)

# transformers용 kwargs
kwargs = quant_config.to_transformers_kwargs()
```

## Health Endpoints

- `/health` - Liveness probe
- `/readiness` - Readiness probe
- `/startup` - Startup probe

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
