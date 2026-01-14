"""
Text Embedder

텍스트 임베딩 생성 모듈
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import os
import logging
import asyncio

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """임베딩 모델"""
    # OpenAI
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"

    # Local / Open Source
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"

    # Ollama
    OLLAMA_NOMIC = "nomic-embed-text"
    OLLAMA_MXBAI = "mxbai-embed-large"


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None


@dataclass
class EmbedderConfig:
    """임베더 설정"""
    model: str = EmbeddingModel.OPENAI_3_SMALL.value
    dimensions: Optional[int] = None  # None이면 모델 기본값
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0


class Embedder(ABC):
    """임베더 베이스 클래스"""

    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """단일 텍스트 임베딩"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 텍스트 임베딩"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """임베딩 차원"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름"""
        pass

    async def embed_with_retry(
        self,
        texts: Union[str, List[str]],
        max_retries: Optional[int] = None
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """재시도 로직이 포함된 임베딩"""
        retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                if isinstance(texts, str):
                    return await self.embed(texts)
                else:
                    return await self.embed_batch(texts)
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise last_error or Exception("Embedding failed")


class OpenAIEmbedder(Embedder):
    """OpenAI 임베딩"""

    DIMENSION_MAP = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, config: Optional[EmbedderConfig] = None, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        """OpenAI 클라이언트 초기화 (lazy loading)"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")

            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required")

            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAI embedder initialized: model={self.config.model}")

    @property
    def dimensions(self) -> int:
        if self.config.dimensions:
            return self.config.dimensions
        return self.DIMENSION_MAP.get(self.config.model, 1536)

    @property
    def model_name(self) -> str:
        return self.config.model

    async def embed(self, text: str) -> EmbeddingResult:
        """단일 텍스트 임베딩"""
        self._ensure_client()

        kwargs = {"model": self.config.model, "input": text}

        # text-embedding-3 모델은 차원 축소 지원
        if self.config.dimensions and "text-embedding-3" in self.config.model:
            kwargs["dimensions"] = self.config.dimensions

        response = await self._client.embeddings.create(**kwargs)

        return EmbeddingResult(
            embedding=response.data[0].embedding,
            model=self.config.model,
            dimensions=len(response.data[0].embedding),
            token_count=response.usage.total_tokens if response.usage else None
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 텍스트 임베딩"""
        self._ensure_client()

        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            kwargs = {"model": self.config.model, "input": batch}
            if self.config.dimensions and "text-embedding-3" in self.config.model:
                kwargs["dimensions"] = self.config.dimensions

            response = await self._client.embeddings.create(**kwargs)

            for j, data in enumerate(response.data):
                results.append(EmbeddingResult(
                    embedding=data.embedding,
                    model=self.config.model,
                    dimensions=len(data.embedding),
                    token_count=None  # 배치에서는 개별 토큰 수 없음
                ))

        return results


class OllamaEmbedder(Embedder):
    """Ollama 임베딩 (로컬)"""

    DIMENSION_MAP = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
    }

    def __init__(
        self,
        config: Optional[EmbedderConfig] = None,
        host: Optional[str] = None
    ):
        super().__init__(config)
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = None

    def _ensure_client(self):
        """Ollama 클라이언트 초기화"""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx is required. Install with: pip install httpx")

            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.config.timeout
            )
            logger.info(f"Ollama embedder initialized: model={self.config.model}, host={self.host}")

    @property
    def dimensions(self) -> int:
        if self.config.dimensions:
            return self.config.dimensions
        return self.DIMENSION_MAP.get(self.config.model, 768)

    @property
    def model_name(self) -> str:
        return self.config.model

    async def embed(self, text: str) -> EmbeddingResult:
        """단일 텍스트 임베딩"""
        self._ensure_client()

        response = await self._client.post(
            "/api/embeddings",
            json={"model": self.config.model, "prompt": text}
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding", [])

        return EmbeddingResult(
            embedding=embedding,
            model=self.config.model,
            dimensions=len(embedding)
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 텍스트 임베딩 (Ollama는 순차 처리)"""
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results


class SentenceTransformersEmbedder(Embedder):
    """Sentence Transformers 임베딩 (로컬)"""

    def __init__(
        self,
        config: Optional[EmbedderConfig] = None,
        model_name: Optional[str] = None
    ):
        super().__init__(config)
        self._model_name = model_name or self.config.model
        self._model = None

    def _ensure_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

            self._model = SentenceTransformer(self._model_name)
            logger.info(f"SentenceTransformers loaded: {self._model_name}")

    @property
    def dimensions(self) -> int:
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> EmbeddingResult:
        """단일 텍스트 임베딩"""
        self._ensure_model()

        # SentenceTransformers는 동기식이므로 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True).tolist()
        )

        return EmbeddingResult(
            embedding=embedding,
            model=self._model_name,
            dimensions=len(embedding)
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """배치 텍스트 임베딩"""
        self._ensure_model()

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=self.config.batch_size
            ).tolist()
        )

        return [
            EmbeddingResult(
                embedding=emb,
                model=self._model_name,
                dimensions=len(emb)
            )
            for emb in embeddings
        ]


def create_embedder(
    model: str = EmbeddingModel.OPENAI_3_SMALL.value,
    **kwargs
) -> Embedder:
    """임베더 팩토리 함수"""
    config = EmbedderConfig(model=model, **{k: v for k, v in kwargs.items() if k in EmbedderConfig.__dataclass_fields__})

    if model.startswith("text-embedding"):
        return OpenAIEmbedder(config, api_key=kwargs.get("api_key"))
    elif model in ["nomic-embed-text", "mxbai-embed-large"]:
        return OllamaEmbedder(config, host=kwargs.get("host"))
    elif model == "sentence-transformers" or "/" in model:
        return SentenceTransformersEmbedder(config, model_name=model)
    else:
        # 기본: OpenAI
        return OpenAIEmbedder(config, api_key=kwargs.get("api_key"))
