# Sample Service

새로운 도메인 서비스 생성을 위한 템플릿입니다.

## 폴더 구조

```
services/sample/
├── sample_service/          # 모든 Python 코드
│   ├── agents/              # 도메인 에이전트
│   ├── tools/               # 도메인 도구
│   ├── integrations/        # 외부 서비스 연동 (선택)
│   ├── api/                 # FastAPI 앱
│   │   ├── routes/          # API 엔드포인트
│   │   ├── middleware/      # 미들웨어
│   │   ├── schemas/         # Request/Response 스키마
│   │   └── main.py          # FastAPI 앱 진입점
│   └── __init__.py
├── config/                  # YAML 설정 파일
├── docker/                  # Docker 설정
├── nginx/                   # Nginx 설정
├── tests/                   # 테스트
└── pyproject.toml
```

## 새 서비스 생성 방법

### 1. 폴더 복사

```bash
cp -r services/sample services/{new-service-name}
cd services/{new-service-name}
```

### 2. 이름 변경

```bash
# 폴더명 변경
mv sample_service {new_service_name}_service

# 파일 내 참조 변경 (sample → {new_service_name})
# - pyproject.toml
# - docker/Dockerfile (uvicorn 경로)
# - docker/docker-compose.yml
# - nginx/sample.conf → nginx/{new_service_name}.conf
# - config/domain.yaml
```

### 3. 필수 수정 파일

| 파일 | 수정 내용 |
|------|----------|
| `pyproject.toml` | name, packages 경로 |
| `config/domain.yaml` | 서비스명, 페르소나, 에이전트 목록 |
| `{name}_service/agents/` | 도메인 에이전트 구현 |
| `{name}_service/tools/` | 도메인 도구 구현 |
| `docker/Dockerfile` | uvicorn 경로 변경 |
| `docker/docker-compose.yml` | DB명, 환경변수 |
| `nginx/{name}.conf` | upstream, server_name |

### 4. 패키지 선택

`pyproject.toml`에서 필요한 패키지 주석 해제:

```toml
[tool.poetry.dependencies]
agentic-ai-core = {path = "../../packages/core", develop = true}

# 지식 그래프 필요 시
agentic-ai-knowledge = {path = "../../packages/knowledge", develop = true}

# 의사결정 지원 필요 시
agentic-ai-decision = {path = "../../packages/decision", develop = true}
```

### 5. 의존성 설치 및 실행

```bash
# 의존성 설치
poetry install

# 개발 서버 실행
uvicorn sample_service.api.main:app --reload --port 8000

# Docker로 실행
cd docker && docker-compose up -d
```

## 설정 파일

| 파일 | 용도 |
|------|------|
| `config/domain.yaml` | 서비스 정의 (이름, 설명, 에이전트) |
| `config/llm.yaml` | LLM 모델 설정 |
| `config/rag.yaml` | RAG 파이프라인 설정 |
| `config/api.yaml` | API 서버 설정 (CORS, 타임아웃) |
| `config/rate_limit.yaml` | Rate limiting 설정 |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/agent/chat` | 동기 채팅 |
| GET | `/agent/chat/stream` | SSE 스트리밍 |
| WS | `/ws/chat/{session_id}` | WebSocket |
| GET | `/health` | 헬스체크 |

## 테스트

```bash
# 전체 테스트
pytest

# 단위 테스트만
pytest tests/unit/

# 통합 테스트만
pytest tests/integration/

# 커버리지
pytest --cov=sample_service
```
