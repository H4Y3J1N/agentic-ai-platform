구현 계획 요약

  MVP (Notion RAG 검색)
  1. ecommerce → internal-ops 폴더 리네이밍
  2. Notion API 클라이언트 구현 (페이지 조회, 블록 추출)
  3. 배치 동기화 매니저 (전체/증분 동기화)
  4. ChromaDB 인덱싱 파이프라인
  5. KnowledgeAgent + NotionSearchTool + NotionPageTool
  6. /knowledge/search, /knowledge/ask API 엔드포인트

  구현 순서: 12단계로 나눔 (폴더 구조 → 모델 → 클라이언트 → 동기화 → 인덱싱 → 에이전트 → API)

  Post-MVP (설계만): Slack 통합, Webhook 실시간 동기화, 제한적 쓰기

● User approved Claude's plan
  ⎿  Plan saved to: ~\.claude\plans\reactive-chasing-thompson.md · /plan to edit
     Internal-Ops Service Implementation Plan

     Overview

     ecommerce 서비스를 internal-ops로 리네이밍하고, Notion RAG 검색 기능을 MVP로 구현합니다.

     MVP Scope

     - Notion 워크스페이스 페이지 인덱싱 (ChromaDB)
     - 자연어 질의응답 (RAG 기반)
     - REST API + SSE 스트리밍

     Post-MVP (설계만, 구현 X)

     - Slack 대화 검색
     - Webhook 실시간 동기화
     - 제한적 쓰기 (페이지 생성/수정, 삭제 불가)

     ---
     Phase 1: Directory Restructure

     1.1 폴더 리네이밍

     services/ecommerce/ → services/internal-ops/
     src/ecommerce_service/ → src/internal_ops_service/

     1.2 새 디렉토리 구조

     services/internal-ops/
     ├── src/internal_ops_service/
     │   ├── agents/
     │   │   └── knowledge_agent.py      # MVP
     │   ├── tools/
     │   │   ├── notion_search_tool.py   # MVP
     │   │   └── notion_page_tool.py     # MVP
     │   └── integrations/
     │       └── notion/
     │           ├── client.py           # API 클라이언트
     │           ├── models.py           # Page, Block 데이터클래스
     │           └── sync.py             # 배치 동기화
     ├── api/
     │   └── routes/
     │       └── knowledge.py            # 새 라우트
     ├── config/
     │   ├── domain.yaml                 # 업데이트
     │   └── notion.yaml                 # 새 파일
     └── scripts/
         └── indexing/
             └── index_notion.py         # CLI 스크립트

     ---
     Phase 2: Notion Integration

     2.1 NotionClient (integrations/notion/client.py)

     - search_pages(): 워크스페이스 페이지 검색
     - get_page(): 페이지 메타데이터
     - get_page_blocks(): 페이지 콘텐츠 (재귀적)
     - iterate_all_pages(): 전체 페이지 순회

     2.2 Models (integrations/notion/models.py)

     - NotionPage: id, title, url, last_edited_time, properties
     - NotionBlock: id, type, content, children
     - RichText: plain_text 추출 헬퍼

     2.3 SyncManager (integrations/notion/sync.py)

     - sync_all(): 전체 동기화
     - sync_incremental(since): 증분 동기화
     - sync_pages(page_ids): 특정 페이지만

     ---
     Phase 3: RAG Pipeline

     3.1 Chunking Strategy

     - 헤더 기반 섹션 분리
     - 코드 블록 보존
     - chunk_size: 500, overlap: 50

     3.2 Metadata 구조

     {
         "notion_page_id": str,
         "notion_url": str,
         "title": str,
         "header_context": str,
         "last_edited_time": str,
         "source": "notion",
         "domain": "internal-ops"
     }

     3.3 ChromaDB Collection

     - name: internal_ops_notion
     - distance: cosine

     ---
     Phase 4: Agent & Tools

     4.1 KnowledgeAgent

     - RAG 검색 후 LLM으로 답변 생성
     - 출처 인용 포함
     - 정보 없으면 명확히 안내

     4.2 NotionSearchTool

     - search(query, top_k, filters): 시맨틱 검색

     4.3 NotionPageTool

     - get_page_content(page_id): 전체 콘텐츠
     - get_page_summary(page_id): 메타데이터만

     ---
     Phase 5: API Routes

     5.1 새 엔드포인트 (/knowledge)

     POST /knowledge/search    # 시맨틱 검색
     POST /knowledge/ask       # Q&A (RAG + LLM)
     GET  /knowledge/page/{id} # 페이지 콘텐츠

     5.2 기존 엔드포인트 유지

     POST /agent/chat          # 일반 채팅
     GET  /agent/chat/stream   # SSE 스트리밍
     WS   /ws/chat/{session}   # WebSocket

     ---
     Phase 6: Config Files

     6.1 notion.yaml

     notion:
       api_key: "${NOTION_API_KEY}"
       sync:
         full_sync_schedule: "0 2 * * *"
         incremental_interval_minutes: 30
         include_archived: false

     6.2 domain.yaml

     domain: "internal-ops"
     agents:
       - knowledge_agent
     capabilities:
       mvp:
         - notion_search
         - knowledge_qa

     ---
     Implementation Order
     ┌───────┬───────────────────┬──────────────────────────────────┐
     │ Phase │       작업        │               파일               │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 1     │ 폴더 리네이밍     │ services/internal-ops/           │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 2     │ Notion 모델       │ integrations/notion/models.py    │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 3     │ Notion 클라이언트 │ integrations/notion/client.py    │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 4     │ Sync 매니저       │ integrations/notion/sync.py      │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 5     │ 인덱싱 스크립트   │ scripts/indexing/index_notion.py │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 6     │ Config 파일들     │ config/notion.yaml, domain.yaml  │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 7     │ NotionSearchTool  │ tools/notion_search_tool.py      │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 8     │ NotionPageTool    │ tools/notion_page_tool.py        │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 9     │ KnowledgeAgent    │ agents/knowledge_agent.py        │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 10    │ API 라우트        │ api/routes/knowledge.py          │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 11    │ main.py 업데이트  │ api/main.py                      │
     ├───────┼───────────────────┼──────────────────────────────────┤
     │ 12    │ 테스트            │ tests/                           │
     └───────┴───────────────────┴──────────────────────────────────┘
     ---
     Verification

     테스트 방법

     1. 인덱싱 테스트
     python scripts/indexing/index_notion.py full
     python scripts/indexing/index_notion.py status
     2. API 테스트
     # 검색
     curl -X POST http://localhost:8000/knowledge/search \
       -H "Content-Type: application/json" \
       -d '{"query": "휴가 정책", "top_k": 5}'

     # Q&A
     curl -X POST "http://localhost:8000/knowledge/ask?question=휴가는 어떻게 신청하나요"
     3. SSE 스트리밍
     curl -N "http://localhost:8000/agent/chat/stream?task=회의록 정리 방법 알려줘"

     성공 기준

     - Notion 페이지 인덱싱 완료 (100+ 페이지)
     - 검색 쿼리 응답 < 2초
     - Q&A 응답에 출처 URL 포함
     - SSE 스트리밍 정상 동작
