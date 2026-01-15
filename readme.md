# 모듈형 Agentic AI 플랫폼

> **핵심 철학**: "AI 엔진은 동일하게 유지하고, 도메인 지식만 교체"

프로덕션 수준의 모듈형 Agentic AI 플랫폼입니다. 재사용 가능한 템플릿으로 설계되어, 플랫폼을 복제하고 YAML 설정만 수정하면 바로 배포할 수 있습니다.

## 주요 기능

- **모듈형 아키텍처**: 명확한 의존성을 가진 6개의 독립 패키지
- **다중 LLM 프로바이더**: OpenAI, Anthropic, 로컬 vLLM 지원
- **RAG 파이프라인**: 청킹, 임베딩, 하이브리드 검색, 품질 평가
- **에이전트 오케스트레이션**: 4가지 패턴 (Supervisor, Hierarchy, Collaborative, Sequential)
- **지식 그래프**: 엔티티 추출, Neo4j 연동
- **로컬 모델 서빙**: vLLM, 멀티테넌트 LoRA, 양자화 (QLoRA, EXL2, GPTQ, AWQ)
- **실시간 통신**: SSE 스트리밍, WebSocket
- **엔터프라이즈 지원**: RBAC, Rate Limiting, 감사 로깅, 관측성

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      클라이언트 레이어                        │
│           웹 앱 / 모바일 앱 / 관리자 API                      │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   API 게이트웨이 (FastAPI) - REST / SSE / WebSocket           │
├─────────────────────────────────────────────────────────────┤
│   보안 레이어 - JWT 인증 / RBAC / Rate Limiting               │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                에이전트 오케스트레이션 레이어                   │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │ Supervisor  │ │  Hierarchy  │ │Collaborative│          │
│   └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │ RAG 엔진    │ │ 지식 그래프  │ │ 의사결정     │          │
│   │             │ │             │ │ 지원        │          │
│   └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      LLM 게이트웨이                           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│   │  OpenAI  │  │Anthropic │  │  vLLM    │                 │
│   └──────────┘  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

## 프로젝트 구조

```
agentic-ai-platform/
├── packages/                    # 재사용 가능한 코어 라이브러리 (레이어 기반)
│   ├── core/                    # 기본 인프라 (LLM, RAG, API)
│   ├── pipeline/                # 데이터 처리 (ingestion, query, scoring, evaluation)
│   ├── agents/                  # 에이전트 오케스트레이션 (orchestrator, base, tools)
│   ├── knowledge/               # 지식 그래프 (entity, graph, ontology)
│   ├── decision/                # 의사결정 지원 (scoring, lifecycle)
│   └── serving/                 # 로컬 모델 서빙 (vLLM, LoRA, quantization)
├── services/                    # 도메인별 서비스
│   ├── sample/                  # 템플릿 (새 서비스 생성 시 복사)
│   └── internal-ops/            # 내부 운영 도구 서비스
└── docs/                        # 문서
```

## 패키지 의존성

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
           serving (선택사항, core에만 의존)
```

## 패키지 선택 가이드

| 사용 사례 | 필요한 패키지 |
|----------|--------------|
| 단순 Q&A 챗봇 | `core` + `agents` |
| 문서 검색 RAG | `core` + `pipeline` + `agents` |
| 지식 관리 시스템 | `core` + `pipeline` + `agents` + `knowledge` |
| 내부 운영 도구 | `core` + `pipeline` + `agents` + `knowledge` + `decision` |
| 로컬 LLM 서빙 | `core` + `serving` |
| 멀티테넌트 LoRA | `core` + `agents` + `serving` |

## 빠른 시작

### 1. 클론 및 설정

```bash
git clone https://github.com/yourusername/agentic-ai-platform.git
cd agentic-ai-platform

# 서비스 의존성 설치
cd services/sample
poetry install
```

### 2. 환경 설정

```bash
cp .env.example .env

# .env 파일에 API 키 입력
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

### 3. 서비스 실행

```bash
# 개발 모드
uvicorn sample_service.api.main:app --reload --port 8000

# 프로덕션 (Docker)
cd docker && docker-compose up -d
```

### 4. API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 채팅 엔드포인트
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"task": "안녕하세요, 무엇을 도와드릴까요?"}'
```

## 새 도메인 서비스 생성

```bash
# 1. 템플릿 복사
cp -r services/sample services/my-service
cd services/my-service
mv sample_service my_service_service

# 2. pyproject.toml, config/domain.yaml 등 수정

# 3. 필요에 따라 패키지 선택
# pyproject.toml 수정:
agentic-core = {path = "../../packages/core", develop = true}
agentic-pipeline = {path = "../../packages/pipeline", develop = true}
agentic-agents = {path = "../../packages/agents", develop = true}
# 필요시 주석 해제:
# agentic-knowledge = {path = "../../packages/knowledge", develop = true}
# agentic-decision = {path = "../../packages/decision", develop = true}
# agentic-serving = {path = "../../packages/serving", develop = true}

# 4. 도메인별 에이전트와 도구 구현
```

## 사용 예시

### RAG 벡터 검색

```python
from agentic_core.rag.stores import MilvusStore

store = MilvusStore(
    collection_name="documents",
    uri="./milvus.db",  # 로컬 파일 (Milvus Lite)
    dimension=1536,
    metric_type="COSINE"
)

await store.insert(ids, texts, embeddings, metadatas)
results = await store.search(query_embedding, top_k=5)
```

### 품질 평가

```python
from agentic_pipeline.evaluation import RAGEvaluator, EvaluationConfig, MetricType

config = EvaluationConfig(
    metrics=[MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCE],
    thresholds={MetricType.FAITHFULNESS: 0.7}
)
evaluator = RAGEvaluator(config=config, llm_judge=my_llm_call)

results = await evaluator.evaluate(
    query="RAG란 무엇인가요?",
    response="RAG는...",
    context=["컨텍스트1", "컨텍스트2"]
)
```

### vLLM + LoRA

```python
from agentic_serving import VLLMProvider, VLLMConfig

config = VLLMConfig(
    base_url="http://localhost:8000/v1",
    default_model="meta-llama/Llama-2-7b-hf",
    enable_lora_routing=True
)

async with VLLMProvider(config) as provider:
    # 기본 추론
    response = await provider.chat(messages)

    # LoRA 어댑터 사용
    response = await provider.chat(messages, lora_adapter="my-adapter")
```

### 멀티테넌트 LoRA

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

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/agent/chat` | 동기 채팅 |
| GET | `/agent/chat/stream` | SSE 스트리밍 |
| WS | `/ws/chat/{session_id}` | WebSocket |
| GET | `/health` | Liveness 프로브 |
| GET | `/readiness` | Readiness 프로브 |

## 기술 스택

- **언어**: Python 3.11+
- **프레임워크**: FastAPI, Poetry
- **데이터베이스**: PostgreSQL, Redis
- **벡터 DB**: ChromaDB, Milvus
- **LLM**: OpenAI, Anthropic, vLLM
- **관측성**: Langfuse, Prometheus
- **배포**: Docker, Kubernetes

## 평가 지표

### RAG 지표
| 지표 | 설명 | 목표값 |
|------|------|--------|
| Faithfulness | 응답이 컨텍스트에 충실한지 | >= 0.7 |
| Answer Relevance | 응답이 질문과 관련 있는지 | >= 0.7 |
| Context Relevance | 검색된 컨텍스트가 관련 있는지 | >= 0.6 |
| Context Recall | 필요한 정보가 컨텍스트에 있는지 | >= 0.6 |

### LLM 지표
| 지표 | 설명 | 목표값 |
|------|------|--------|
| Coherence | 논리적 일관성 | >= 0.7 |
| Fluency | 문법적 유창성 | >= 0.7 |
| Toxicity | 유해성 (낮을수록 좋음) | <= 0.1 |
| Hallucination | 환각 (낮을수록 좋음) | <= 0.2 |

## 양자화 지원

| 방식 | 용도 | 속도 | 품질 | VRAM 절감 |
|------|------|------|------|-----------|
| QLoRA | 파인튜닝 | 중간 | 높음 | ~75% |
| EXL2 | 추론 | 빠름 | 높음 | ~75% |
| GPTQ | 추론 | 빠름 | 중간 | ~75% |
| AWQ | 추론 | 빠름 | 높음 | ~75% |
| FP8 | 추론 (vLLM) | 빠름 | 매우 높음 | ~50% |

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
