# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular Agentic AI platform designed as a reusable template. The core philosophy: "AI engine stays the same, only swap out domain knowledge." Clone the platform, configure domain-specific settings in YAML, and deploy.

**Tech Stack**: Python 3.11+, FastAPI, Poetry, PostgreSQL, Redis, ChromaDB (vector DB), Docker

## Common Commands

### Development
```bash
# Start the e-commerce service (from repo root)
cd services/ecommerce
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Docker development (simple)
cd services/ecommerce/docker
docker-compose up -d

# Full stack with ChromaDB
docker-compose -f infrastructure/docker/docker-compose.base.yml up -d
```

### Testing
```bash
# Run tests (from services/ecommerce)
pytest tests/

# Run specific test file
pytest tests/unit/test_rate_limiter.py
pytest tests/integration/test_sse_streaming.py

# Manual endpoint tests
curl -N http://localhost:8000/agent/chat/stream?task="test"  # SSE
python scripts/monitoring/test_websocket.py                   # WebSocket
```

## Architecture

```
packages/agentic-ai-core/     # Reusable core library (install as dependency)
├── agents/                   # Base agent classes, registry, streaming
├── api/                      # FastAPI components (SSE, WebSocket, rate limiting)
├── llm/                      # LLM gateway with multi-provider support (OpenAI, Anthropic, local)
├── orchestrator/             # Orchestration patterns: supervisor, hierarchy, collaborative, sequential
├── rag/                      # RAG pipeline (embedder, retriever, chunker, chroma_store)
├── security/                 # RBAC, JWT, token bucket rate limiting
├── observability/            # Langfuse integration for LLM tracking
└── tools/                    # Base tool class, registry, security wrapper

services/ecommerce/           # Domain-specific service (example implementation)
├── src/ecommerce_service/    # Domain agents and tools
├── api/                      # FastAPI app with routes, middleware, WebSocket handlers
├── config/                   # YAML configs (api, rag, rate_limit, domain, websocket)
├── knowledge/docs/           # Domain knowledge for RAG
└── tests/                    # Unit and integration tests

infrastructure/
├── docker/                   # docker-compose.base.yml (PostgreSQL, Redis, optional Ollama)
└── nginx/                    # Nginx proxy configuration (for production)
```

## Key Patterns

**Orchestration**: Four patterns in `packages/agentic-ai-core/src/agentic_ai_core/orchestrator/patterns/`:
- `supervisor.py`: Main supervisor delegates to specialist agents
- `hierarchy.py`: Multi-level delegation (L1 -> L2 -> L3)
- `collaborative.py`: Multiple agents work simultaneously, merge results
- `sequential.py`: Pipeline where each agent's output feeds the next

**API Communication**:
- REST: `POST /agent/chat`
- SSE Streaming: `GET /agent/chat/stream` (real-time token streaming)
- WebSocket: `WS /ws/chat/{session_id}` (bidirectional)

**Rate Limiting**: Token bucket (in-memory). Configured per-endpoint and per-role in `config/rate_limit.yaml`.

**Observability**: Langfuse integration for LLM call tracking, cost monitoring, and latency analysis.

## Configuration

All major components use YAML configuration in `services/{domain}/config/`:
- `domain.yaml`: Domain name, persona, RAG sources, agent list
- `api.yaml`: CORS settings, timeouts
- `rate_limit.yaml`: Request limits by endpoint and role
- `rag.yaml`: Embedding model, chunk size, retrieval settings
- `llm.yaml`: Provider settings, routing strategy

Environment variables for external services:
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
```

## Creating a New Domain Service

1. Copy `services/ecommerce/` as template
2. Implement domain agents in `src/{domain}_service/agents/`
3. Implement domain tools in `src/{domain}_service/tools/`
4. Add domain knowledge to `knowledge/docs/`
5. Update `config/domain.yaml` with domain-specific settings

## Deployment

### Single Server (Recommended for SMB)
```bash
docker-compose -f infrastructure/docker/docker-compose.base.yml up -d
```

Health endpoints for load balancer:
- `/health` - liveness check
- `/readiness` - readiness check
- `/startup` - startup check

Graceful shutdown with 30s connection draining is built into `api/main.py`.

### Scaling Up (Future)

When scaling beyond single server, consider adding:
- Nginx reverse proxy
- K8s with k3s (lightweight)
- Milvus instead of ChromaDB for large vector datasets
- Redis distributed rate limiting

## Data Layer

- **PostgreSQL**: Business data (orders, customers, products)
- **ChromaDB**: Vector embeddings for RAG (file-based, lightweight)
- **Redis**: Cache and sessions
