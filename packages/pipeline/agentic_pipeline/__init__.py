"""
Agentic Pipeline Package

데이터 처리 파이프라인
- ingestion: 데이터 수집 및 파싱
- query: 쿼리 처리 및 하이브리드 검색
- scoring: 관련성 스코어링, ROI 계산
- evaluation: RAG/LLM 품질 평가
"""

__version__ = "0.1.0"

# Submodule imports (lazy loading을 위해 모듈만 import)
from . import ingestion
from . import query
from . import scoring
from . import evaluation

__all__ = [
    "ingestion",
    "query",
    "scoring",
    "evaluation",
]
