# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular Agentic AI platform designed as a reusable template. The core philosophy: "AI engine stays the same, only swap out domain knowledge." Clone the platform, configure domain-specific settings in YAML, and deploy.

**Tech Stack**: Python 3.11+, FastAPI, Poetry, PostgreSQL, Redis, ChromaDB (vector DB), Langfuse, Docker

## Current Development Status

**Active Service**: `internal-ops` (renaming from `ecommerce`)

**Purpose**: Internal efficiency tool with Slack/Notion integration, designed for future B2B and government projects.

**MVP Scope**: Notion RAG search (knowledge retrieval from Notion documents)

**Planned Features**:
- Notion document indexing and natural language Q&A
- Slack conversation search
- Hybrid sync (batch + webhook for important events)
- Limited write operations (create/edit Notion pages, reply to Slack - NO delete)
- Task automation via Slack commands

## Common Commands

### Development
```bash
# Start the internal-ops service (from repo root)
cd services/internal-ops
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Docker development (simple)
cd services/internal-ops/docker
docker-compose up -d

# Full stack
docker-compose -f infrastructure/docker/docker-compose.base.yml up -d
```

### Testing
```bash
# Run tests (from services/internal-ops)
pytest tests/

# Run specific test file
pytest tests/unit/test_rate_limiter.py
pytest tests/integration/test_sse_streaming.py
```

### Notion Indexing (after implementation)
```bash
# Index Notion workspace
python scripts/indexing/index_notion.py --workspace-id <WORKSPACE_ID>
```

## Architecture

```
packages/agentic-ai-core/        # Reusable core library
├── agents/                      # Base agent classes, registry, streaming
├── api/                         # FastAPI components (SSE, WebSocket, rate limiting)
├── llm/                         # LLM gateway (OpenAI, Anthropic, local Ollama)
├── orchestrator/                # Patterns: supervisor, hierarchy, collaborative, sequential
├── rag/                         # RAG pipeline (embedder, retriever, chunker, chroma_store)
├── security/                    # RBAC, JWT, token bucket rate limiting
├── observability/               # Langfuse integration
└── tools/                       # Base tool class, registry, security wrapper

services/internal-ops/           # Internal operations service
├── src/internal_ops_service/
│   ├── agents/                  # KnowledgeAgent, AutomationAgent
│   ├── tools/                   # NotionSearchTool, NotionWriteTool, SlackTools
│   └── integrations/            # Notion/Slack API clients
├── api/                         # FastAPI app, routes, middleware
├── config/                      # YAML configs (domain, notion, slack, rag, etc.)
├── knowledge/                   # Local knowledge docs (if any)
└── tests/

infrastructure/
├── docker/                      # docker-compose.base.yml
└── nginx/                       # Nginx proxy (production)
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

**Observability**: Langfuse for LLM call tracking, cost monitoring, and latency analysis.

## Configuration

YAML configs in `services/internal-ops/config/`:
- `domain.yaml`: Service name, persona, agent list
- `notion.yaml`: Notion API settings, sync config
- `slack.yaml`: Slack API settings (future)
- `api.yaml`: CORS, timeouts
- `rate_limit.yaml`: Request limits by endpoint/role
- `rag.yaml`: Embedding model, chunk size, retrieval settings

Environment variables:
```bash
# LLM Providers
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Observability
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com

# Integrations
NOTION_API_KEY=...
NOTION_WORKSPACE_ID=...
SLACK_BOT_TOKEN=...        # future
SLACK_SIGNING_SECRET=...   # future
```

## Data Flow (Notion RAG)

```
1. Indexing Pipeline:
   Notion API → Fetch pages → Chunk text → Embed → Store in ChromaDB

2. Query Pipeline:
   User query → Embed query → ChromaDB similarity search → Context retrieval
   → LLM generates answer with context → Response

3. Sync Strategy (Hybrid):
   - Batch: Scheduled full/incremental sync (e.g., hourly)
   - Webhook: Real-time updates for critical changes (future)
```

## Creating a New Domain Service

1. Copy `services/internal-ops/` as template
2. Rename to `services/{new-domain}/`
3. Update `src/{new_domain}_service/` module name
4. Implement domain-specific agents and tools
5. Configure `config/domain.yaml`

## Deployment

### Single Server (Current)
```bash
docker-compose -f infrastructure/docker/docker-compose.base.yml up -d
```

Health endpoints: `/health`, `/readiness`, `/startup`

### Scaling Up (Future)
- Nginx reverse proxy
- K8s with k3s
- Milvus for large vector datasets
- Redis distributed rate limiting

## Data Layer

- **PostgreSQL**: User data, sync state, audit logs
- **ChromaDB**: Vector embeddings for RAG
- **Redis**: Cache and sessions
