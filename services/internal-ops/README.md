  services/internal-ops/                    # ecommerce에서 리네이밍
  ├── src/internal_ops_service/
  │   ├── agents/
  │   │   ├── __init__.py
  │   │   └── knowledge_agent.py           # RAG 기반 Q&A
  │   ├── tools/
  │   │   ├── __init__.py
  │   │   ├── notion_search_tool.py        # 시맨틱 검색
  │   │   └── notion_page_tool.py          # 페이지 조회
  │   └── integrations/notion/
  │       ├── __init__.py
  │       ├── client.py                    # Notion API 클라이언트
  │       ├── models.py                    # NotionPage, NotionBlock
  │       └── sync.py                      # 배치 동기화
  ├── api/
  │   ├── main.py                          # 업데이트됨
  │   ├── routes/
  │   │   └── knowledge.py                 # 새 API 라우트
  │   └── schemas/
  │       └── knowledge.py                 # Pydantic 스키마
  ├── config/
  │   ├── domain.yaml                      # 서비스 설정
  │   ├── notion.yaml                      # Notion 설정
  │   └── rag.yaml                         # RAG 설정
  └── scripts/indexing/
      └── index_notion.py                  # 인덱싱 CLI