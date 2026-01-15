# Modular Agentic AI Platform

> **Core Philosophy**: "AI engine stays the same, only swap out domain knowledge"

A production-ready, modular Agentic AI platform designed as a reusable template. Clone the platform, configure domain-specific settings in YAML, and deploy.

## Features

- **Modular Architecture**: 6 independent packages with clear dependencies
- **Multiple LLM Providers**: OpenAI, Anthropic, local vLLM support
- **RAG Pipeline**: Chunking, embedding, hybrid search, quality evaluation
- **Agent Orchestration**: 4 patterns (Supervisor, Hierarchy, Collaborative, Sequential)
- **Knowledge Graph**: Entity extraction, Neo4j integration
- **Local Model Serving**: vLLM, multi-tenant LoRA, quantization (QLoRA, EXL2, GPTQ, AWQ)
- **Real-time Communication**: SSE streaming, WebSocket
- **Enterprise Ready**: RBAC, rate limiting, audit logging, observability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│         Web App / Mobile App / Admin API                     │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   API Gateway (FastAPI) - REST / SSE / WebSocket             │
├─────────────────────────────────────────────────────────────┤
│   Security Layer - JWT Auth / RBAC / Rate Limiting           │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Agent Orchestration Layer                       │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │ Supervisor  │ │  Hierarchy  │ │Collaborative│          │
│   └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │ RAG Engine  │ │Knowledge    │ │  Decision   │          │
│   │             │ │Graph        │ │  Support    │          │
│   └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    LLM Gateway                               │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│   │  OpenAI  │  │Anthropic │  │  vLLM    │                 │
│   └──────────┘  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
agentic-ai-platform/
├── packages/                    # Reusable core libraries (layer-based)
│   ├── core/                    # Base infrastructure (LLM, RAG, API)
│   ├── pipeline/                # Data processing (ingestion, query, scoring, evaluation)
│   ├── agents/                  # Agent orchestration (orchestrator, base, tools)
│   ├── knowledge/               # Knowledge graph (entity, graph, ontology)
│   ├── decision/                # Decision support (scoring, lifecycle)
│   └── serving/                 # Local model serving (vLLM, LoRA, quantization)
├── services/                    # Domain-specific services
│   ├── sample/                  # Template (copy this to create new services)
│   └── internal-ops/            # Internal operations service
└── docs/                        # Documentation
```

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

## Package Selection Guide

| Use Case | Required Packages |
|----------|-------------------|
| Simple Q&A Chatbot | `core` + `agents` |
| Document Search RAG | `core` + `pipeline` + `agents` |
| Knowledge Management | `core` + `pipeline` + `agents` + `knowledge` |
| Internal Operations | `core` + `pipeline` + `agents` + `knowledge` + `decision` |
| Local LLM Serving | `core` + `serving` |
| Multi-tenant LoRA | `core` + `agents` + `serving` |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/agentic-ai-platform.git
cd agentic-ai-platform

# Install dependencies for a service
cd services/sample
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

### 3. Run the Service

```bash
# Development
uvicorn sample_service.api.main:app --reload --port 8000

# Production (Docker)
cd docker && docker-compose up -d
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"task": "Hello, how can you help me?"}'
```

## Creating a New Domain Service

```bash
# 1. Copy the template
cp -r services/sample services/my-service
cd services/my-service
mv sample_service my_service_service

# 2. Update pyproject.toml, config/domain.yaml, etc.

# 3. Choose packages based on your needs
# Edit pyproject.toml:
agentic-core = {path = "../../packages/core", develop = true}
agentic-pipeline = {path = "../../packages/pipeline", develop = true}
agentic-agents = {path = "../../packages/agents", develop = true}
# Uncomment as needed:
# agentic-knowledge = {path = "../../packages/knowledge", develop = true}
# agentic-decision = {path = "../../packages/decision", develop = true}
# agentic-serving = {path = "../../packages/serving", develop = true}

# 4. Implement domain-specific agents and tools
```

## Usage Examples

### RAG with Vector Search

```python
from agentic_core.rag.stores import MilvusStore

store = MilvusStore(
    collection_name="documents",
    uri="./milvus.db",  # Local file (Milvus Lite)
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
    query="What is RAG?",
    response="RAG stands for...",
    context=["context1", "context2"]
)
```

### vLLM with LoRA

```python
from agentic_serving import VLLMProvider, VLLMConfig

config = VLLMConfig(
    base_url="http://localhost:8000/v1",
    default_model="meta-llama/Llama-2-7b-hf",
    enable_lora_routing=True
)

async with VLLMProvider(config) as provider:
    # Standard inference
    response = await provider.chat(messages)

    # With LoRA adapter
    response = await provider.chat(messages, lora_adapter="my-adapter")
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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/chat` | Synchronous chat |
| GET | `/agent/chat/stream` | SSE streaming |
| WS | `/ws/chat/{session_id}` | WebSocket |
| GET | `/health` | Liveness probe |
| GET | `/readiness` | Readiness probe |

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI, Poetry
- **Database**: PostgreSQL, Redis
- **Vector DB**: ChromaDB, Milvus
- **LLM**: OpenAI, Anthropic, vLLM
- **Observability**: Langfuse, Prometheus
- **Deployment**: Docker, Kubernetes

## Evaluation Metrics

### RAG Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Response grounded in context | >= 0.7 |
| Answer Relevance | Response relevant to query | >= 0.7 |
| Context Relevance | Retrieved context relevant | >= 0.6 |
| Context Recall | Required info in context | >= 0.6 |

### LLM Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Coherence | Logical consistency | >= 0.7 |
| Fluency | Grammatical quality | >= 0.7 |
| Toxicity | Harmful content (lower is better) | <= 0.1 |
| Hallucination | Factual errors (lower is better) | <= 0.2 |

## Quantization Support

| Method | Use Case | Speed | Quality | VRAM Saving |
|--------|----------|-------|---------|-------------|
| QLoRA | Fine-tuning | Medium | High | ~75% |
| EXL2 | Inference | Fast | High | ~75% |
| GPTQ | Inference | Fast | Medium | ~75% |
| AWQ | Inference | Fast | High | ~75% |
| FP8 | Inference (vLLM) | Fast | Very High | ~50% |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
