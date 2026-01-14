# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular Agentic AI platform designed as a reusable template. Core philosophy: "AI engine stays the same, only swap out domain knowledge." Clone the platform, configure domain-specific settings in YAML, and deploy.

**Tech Stack**: Python 3.11+, FastAPI, Poetry, PostgreSQL, Redis, ChromaDB, Langfuse, Docker

## Project Structure

```
agentic-ai-platform/
├── packages/                    # 재사용 가능한 코어 라이브러리
│   ├── core/                    # 필수 - 모든 도메인에서 사용
│   ├── knowledge/               # 선택 - 지식그래프 필요 시
│   └── decision/                # 선택 - 의사결정 지원 필요 시
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
```

## Architecture

### packages/core (공통)

```
packages/core/agentic_ai_core/
├── llm/             # LLM gateway (OpenAI, Anthropic, Ollama)
├── rag/             # Vector search: chunker, embedder, retriever, stores/
├── ingestion/       # Data pipeline: Pipeline, Stages
├── query/           # Hybrid search, query rewriting, fusion
├── scoring/         # Relevance scoring, ROI calculation
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

Environment variables:
```bash
OPENAI_API_KEY, ANTHROPIC_API_KEY          # LLM providers
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY   # Observability
NOTION_API_KEY, NOTION_WORKSPACE_ID        # Integrations (internal-ops)
DATABASE_URL, REDIS_URL, CHROMA_PERSIST_DIR
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
```

4. Implement domain logic:
   - `{name}_service/agents/` - Domain agents
   - `{name}_service/tools/` - Domain tools

See `services/sample/README.md` for detailed guide.

## Health Endpoints

- `/health` - Liveness probe
- `/readiness` - Readiness probe
- `/startup` - Startup probe
