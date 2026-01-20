 Internal-Ops Chatbot Implementation Plan

 Overview

 사내 Notion/Slack 데이터를 활용한 AI 챗봇 구현. 업무 진행상황 질의 및 태스크 생성 기능 제공.

 Database Structure

 [Project DB] (d88921b8328f48f0b1b07267afffe193)
 ├── 제목 (Title)
 ├── 담당 국가 (Relation → Country DB)
 ├── 사업 스테이지 (Select)
 ├── 담당자 (Person)
 ├── 관련 Task (Relation → Task DB)  ← 여기에 태스크 추가
 └── 관련 미팅 (Relation → Meeting DB)

 ---
 Phase 1: Task Creation Tool (Priority 1)

 1.1 Notion Client 수정

 File: internal_ops_service/integrations/notion/client.py

 create_page() 메서드에 properties 파라미터 추가:
 async def create_page(
     self,
     parent_id: str,
     parent_type: str,
     title: str,
     properties: Optional[Dict[str, Any]] = None,  # NEW
     content_blocks: Optional[List[Dict]] = None
 ) -> NotionPage

 1.2 Task Creation Tool 생성

 File: internal_ops_service/tools/task_creation_tool.py

 class TaskCreationTool:
     async def search_projects(query: str, limit: int = 5) -> List[Dict]
     async def create_task(title: str, project_id: str = None, project_name: str = None) -> Dict

 1.3 Configuration 추가

 File: config/notion.yaml

 notion:
   databases:
     project: "d88921b8328f48f0b1b07267afffe193"
     task: "${NOTION_TASK_DB_ID}"
   task_creation:
     task_project_relation_property: "프로젝트"  # Task DB에서 Project를 가리키는 Relation 속성명

 1.4 API Routes

 File: internal_ops_service/api/routes/tasks.py

 - POST /tasks/create - 태스크 생성
 - GET /tasks/projects/search - 프로젝트 검색 (링크용)

 File: internal_ops_service/api/schemas/tasks.py

 - TaskCreateRequest, TaskCreateResponse, ProjectSearchResponse

 ---
 Phase 2: Slack Integration

 2.1 Slack Client

 File: internal_ops_service/integrations/slack/client.py

 class SlackClient:
     async def list_channels() -> AsyncIterator[Dict]
     async def get_channel_history(channel_id, oldest, latest) -> AsyncIterator[Dict]
     async def get_thread_replies(channel_id, thread_ts) -> List[Dict]
     async def get_user_info(user_id) -> Dict

 Uses slack-sdk (AsyncWebClient)

 2.2 Slack Models

 File: internal_ops_service/integrations/slack/models.py

 - SlackMessage, SlackChannel, SlackUser dataclasses

 2.3 Slack Sync Manager

 File: internal_ops_service/integrations/slack/sync.py

 class SlackSyncManager:
     async def sync_all() -> AsyncIterator[SyncedSlackDocument]
     async def sync_incremental(since: datetime) -> AsyncIterator[SyncedSlackDocument]

 2.4 Slack Search Tool

 File: internal_ops_service/tools/slack_search_tool.py

 class SlackSearchTool:
     async def search(query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]

 ChromaDB + OpenAI embeddings (NotionSearchTool과 동일 패턴)

 2.5 Configuration

 File: config/slack.yaml

 slack:
   bot_token: "${SLACK_BOT_TOKEN}"
   workspace_url: "${SLACK_WORKSPACE_URL}"
   sync:
     include_threads: true
     lookback_days: 90

 2.6 Indexing Script

 File: scripts/index_slack.py

 python index_slack.py full        # 전체 동기화
 python index_slack.py incremental # 증분 동기화

 ---
 Phase 3: Unified Agent

 3.1 Internal Ops Agent

 File: internal_ops_service/agents/internal_ops_agent.py

 class InternalOpsAgent:
     """Orchestrates Notion search, Slack search, Task creation"""

     def _detect_intent(task: str) -> str  # "search" | "create_task" | "ask"
     async def execute(task: str, context: Dict) -> str
     async def execute_stream(task: str, context: Dict) -> AsyncIterator[str]

 Intent-based routing:
 - "태스크 만들어줘" → TaskCreationTool
 - "검색해줘" → Unified search (Notion + Slack)
 - 일반 질문 → KnowledgeAgent (기존 RAG)

 3.2 Module Exports 업데이트

 File: internal_ops_service/tools/__init__.py
 from .task_creation_tool import TaskCreationTool
 from .slack_search_tool import SlackSearchTool

 File: internal_ops_service/agents/__init__.py
 from .internal_ops_agent import InternalOpsAgent

 File: internal_ops_service/integrations/slack/__init__.py
 from .client import SlackClient, SlackClientConfig
 from .models import SlackMessage, SlackChannel, SlackUser
 from .sync import SlackSyncManager, SlackSyncConfig

 ---
 Files Summary

 New Files (10)

 internal_ops_service/
 ├── integrations/slack/
 │   ├── __init__.py
 │   ├── client.py
 │   ├── models.py
 │   └── sync.py
 ├── tools/
 │   ├── task_creation_tool.py
 │   └── slack_search_tool.py
 ├── agents/
 │   └── internal_ops_agent.py
 ├── api/
 │   ├── routes/tasks.py
 │   └── schemas/tasks.py
 config/
 └── slack.yaml
 scripts/
 └── index_slack.py

 Modified Files (5)

 internal_ops_service/integrations/notion/client.py  # create_page() 수정
 internal_ops_service/tools/__init__.py              # exports 추가
 internal_ops_service/agents/__init__.py             # exports 추가
 internal_ops_service/api/main.py                    # tasks router 추가
 config/notion.yaml                                  # database IDs 추가
 pyproject.toml                                      # slack-sdk 의존성 추가

 ---
 Environment Variables

 # Existing
 NOTION_API_KEY=secret_xxx
 OPENAI_API_KEY=sk-xxx

 # New
 NOTION_TASK_DB_ID=xxx          # Task DB ID (또는 동적 탐색)
 SLACK_BOT_TOKEN=xoxb-xxx       # Slack Bot Token
 SLACK_WORKSPACE_URL=https://xxx.slack.com

 ---
 Verification

 Phase 1 검증

 # 1. 서버 실행
 uvicorn internal_ops_service.api.main:app --reload --port 8000

 # 2. 프로젝트 검색 테스트
 curl "http://localhost:8000/tasks/projects/search?query=일본"

 # 3. 태스크 생성 테스트
 curl -X POST http://localhost:8000/tasks/create \
   -H "Content-Type: application/json" \
   -d '{"title": "테스트 태스크", "project_name": "프로젝트명"}'

 # 4. Notion에서 생성된 태스크 확인

 Phase 2 검증

 # 1. Slack 인덱싱
 python scripts/index_slack.py full

 # 2. Slack 검색 테스트
 curl -X POST http://localhost:8000/knowledge/search \
   -H "Content-Type: application/json" \
   -d '{"query": "미팅 일정", "source": "slack"}'

 Phase 3 검증

 # 통합 채팅 테스트
 curl -X POST http://localhost:8000/agent/chat \
   -H "Content-Type: application/json" \
   -d '{"task": "일본 프로젝트 진행상황 알려줘"}'

 curl -X POST http://localhost:8000/agent/chat \
   -H "Content-Type: application/json" \
   -d '{"task": "프로젝트 A에 계약서 검토 태스크 추가해줘"}'

 ---
 Implementation Order

 1. Phase 1 (Task Creation)
   - NotionClient.create_page() 수정
   - TaskCreationTool 구현
   - API routes/schemas
   - 테스트
 2. Phase 2 (Slack)
   - Slack integration 모듈
   - SlackSearchTool
   - 인덱싱 스크립트
   - 테스트
 3. Phase 3 (Unified Agent)
   - InternalOpsAgent 구현
   - Intent detection
   - 통합 테스트


한편으로는, 구현된 후에 chunking, indexing, metadata 등 더 세분화한 성능 고도화 필요.
langfuse 도 적용해보고 싶고. 

그리고 llm 은 google의 제미나이를 사용하고 싶어. 외부 api 사용. 그리고 vllm 을 사용해보고 싶어. continuous batching 적용해보고 싶거든.

streamlit 과 sse 를 적용해서, 웹으로 채팅을 주고받을 수 있으면 좋겠어.  근데 이건 C:\Users\peter\Downloads\agentic-ai-platform-complete\packages 에 범용으로 사용할 수 있는 걸 추가하는 게 범용적이고 좋으려나?                                                                            ''



 Internal-Ops 서비스 아키텍처                                                                                                                                    
                                                                                                                                                                                                                                                                                             
  ┌──────────────────────────────────────────────────────────────────────────────┐                                                                                                                                                                                                                                   EXTERNAL CLIENTS                                 │
  │                     (Web App, CLI, Slack Bot, etc.)                          │                                                                                                                                                                                                             └─────────────────────────────────┬────────────────────────────────────────────┘                                                                                                                                                                                                     
                                    │ HTTP / WebSocket
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                          API LAYER (FastAPI)                                 │
  │  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐    │
  │  │  /agent/*   │ │ /knowledge/*│ │  /tasks/*    │ │ /ws/chat/{session}  │    │
  │  │  chat.py    │ │ knowledge.py│ │  tasks.py    │ │   websocket.py      │    │
  │  └─────────────┘ └─────────────┘ └──────────────┘ └─────────────────────┘    │
  │                        Middleware: Auth, CORS, RateLimit                     │
  └─────────────────────────────────┬────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                           AGENT LAYER                                        │
  │  ┌──────────────────────────────────────────────────────────────────────┐    │
  │  │                    InternalOpsAgent (Orchestrator)                    │   │
  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │   │
  │  │  │create_task │  │search_slack│  │search_notion│ │ unified_search │   │   │
  │  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘   │   │
  │  │                         Intent Detection (NLP)                        │   │
  │  └──────────────────────────────────────────────────────────────────────┘    │
  │                                    │                                         │
  │  ┌──────────────────────────────────────────────────────────────────────┐    │
  │  │                      KnowledgeAgent (RAG Specialist)                  │   │
  │  │               Retrieval → Context Building → Generation               │   │
  │  └──────────────────────────────────────────────────────────────────────┘    │
  └─────────────────────────────────┬────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                            TOOLS LAYER                                       │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
  │  │ NotionSearchTool│  │ SlackSearchTool │  │    TaskCreationTool         │   │
  │  │   (RAG Search)  │  │   (RAG Search)  │  │ (Create + Link to Project)  │   │
  │  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘   │
  │           │                    │                          │                  │
  │           └────────────────────┼──────────────────────────┘                  │
  │                                │                                             │
  └────────────────────────────────┼─────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
  ┌───────────────┐      ┌─────────────────┐      ┌─────────────────────┐
  │  INTEGRATIONS │      │   VECTOR DB     │      │     LLM GATEWAY     │
  │               │      │                 │      │                     │
  │ ┌───────────┐ │      │   ┌─────────┐   │      │  ┌───────────────┐  │
  │ │  Notion   │ │      │   │ChromaDB │   │      │  │   LiteLLM     │  │
  │ │  Client   │◄├──────┤   │         │   │      │  │  (100+ LLMs)  │  │
  │ └───────────┘ │      │   │ notion  │   │      │  └───────┬───────┘  │
  │ ┌───────────┐ │      │   │ slack   │   │      │          │          │
  │ │  Slack    │ │      │   │collections  │      │          ▼          │
  │ │  Client   │◄├──────┤   └─────────┘   │      │  ┌───────────────┐  │
  │ └───────────┘ │      │                 │      │  │    Gemini     │  │
  └───────────────┘      └─────────────────┘      │  │  (default)    │  │
          │                      │                │  ├───────────────┤  │
          │                      │                │  │    OpenAI     │  │
          ▼                      ▼                │  ├───────────────┤  │
  ┌───────────────┐      ┌─────────────────┐      │  │    Claude     │  │
  │ EXTERNAL APIs │      │   EMBEDDINGS    │      │  ├───────────────┤  │
  │               │      │                 │      │  │  vLLM (self)  │  │
  │ • Notion API  │      │  OpenAI         │      │  └───────────────┘  │
  │ • Slack API   │      │  text-embedding │      └─────────────────────┘
  │               │      │  -3-small       │
  └───────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                                                ┌─────────────────────┐
                                                │   OBSERVABILITY     │
                                                │                     │
                                                │  ┌───────────────┐  │
                                                │  │   Langfuse    │  │
                                                │  │   Tracing     │  │
                                                │  └───────────────┘  │
                                                └─────────────────────┘

  ---
  패키지 의존성 계층

  ┌─────────────────────────────────────────────────────────────────┐
  │                    services/internal-ops                         │
  │                    (internal_ops_service)                        │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │ depends on
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  agentic-core   │  │ agentic-agents  │  │agentic-knowledge│
  │                 │  │                 │  │                 │
  │ • LLM Gateway   │  │ • Base Agent    │  │ • RAG Pipeline  │
  │ • RAG (Chunker, │  │ • Tool Protocol │  │ • Retrieval     │
  │   Embedder,     │  │                 │  │                 │
  │   Indexer)      │  │                 │  │                 │
  │ • Observability │  │                 │  │                 │
  │   (Langfuse)    │  │                 │  │                 │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                    │
           └────────────────────┴────────────────────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │ agentic-pipeline│
                      │                 │
                      │ • Flow Control  │
                      │ • Step Execution│
                      └─────────────────┘

  ---
  데이터 흐름

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                            INDEXING (Offline)                               │
  │                                                                             │
  │  ┌─────────┐    index_notion.py     ┌─────────┐    embed     ┌──────────┐  │
  │  │ Notion  │ ───────────────────►   │ Chunker │ ─────────►   │ ChromaDB │  │
  │  │   API   │    (RecursiveChunker)  │  (Core) │   (OpenAI)   │ (notion) │  │
  │  └─────────┘                        └─────────┘              └──────────┘  │
  │                                                                            │
  │  ┌─────────┐    index_slack.py      ┌─────────┐    embed     ┌──────────┐  │
  │  │  Slack  │ ───────────────────►   │ Chunker │ ─────────►   │ ChromaDB │  │
  │  │   API   │    (RecursiveChunker)  │  (Core) │   (OpenAI)   │ (slack)  │  │
  │  └─────────┘                        └─────────┘              └──────────┘  │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           QUERY (Online)                                    │
  │                                                                             │
  │  User Query                                                                 │
  │      │                                                                      │
  │      ▼                                                                      │
  │  ┌────────────┐    embed      ┌──────────┐   search    ┌─────────────┐     │
  │  │   Query    │ ──────────►   │ ChromaDB │ ─────────►  │  Top-K Docs │     │
  │  │            │   (OpenAI)    │          │  (cosine)   │  + metadata │     │
  │  └────────────┘               └──────────┘             └──────┬──────┘     │
  │                                                              │             │
  │                                                              ▼             │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │                        LLM Generation                                │  │
  │  │                                                                      │  │
  │  │   System Prompt + Retrieved Context + User Question                  │  │
  │  │                          │                                           │  │
  │  │                          ▼                                           │  │
  │  │                  ┌──────────────┐                                    │  │
  │  │                  │   Gemini     │                                    │  │
  │  │                  │ 1.5 Flash    │                                    │  │
  │  │                  └──────────────┘                                    │  │
  │  │                          │                                           │  │
  │  │                          ▼                                           │  │
  │  │              Answer with Citations [1], [2]                          │  │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────────┘

  ---
  환경 변수 요약
  ┌─────────────────────┬──────┬──────────────────────────────────────────┐
  │        변수         │ 필수 │                   설명                   │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ NOTION_API_KEY      │ ✅   │ Notion API 키                            │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ OPENAI_API_KEY      │ ✅   │ 임베딩용 (text-embedding-3-small)        │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ GEMINI_API_KEY      │ ✅   │ LLM 생성용 (기본 모델)                   │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ SLACK_BOT_TOKEN     │ ✅   │ Slack 봇 토큰                            │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ SLACK_USER_TOKEN    │ ❌   │ Slack 검색 API용 (선택)                  │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ LANGFUSE_PUBLIC_KEY │ ❌   │ 관측성 (선택)                            │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ LANGFUSE_SECRET_KEY │ ❌   │ 관측성 (선택)                            │
  ├─────────────────────┼──────┼──────────────────────────────────────────┤
  │ CHROMA_PERSIST_DIR  │ ❌   │ ChromaDB 저장 경로 (기본: ./data/chroma) │
  └─────────────────────┴──────┴──────────────────────────────────────────┘

