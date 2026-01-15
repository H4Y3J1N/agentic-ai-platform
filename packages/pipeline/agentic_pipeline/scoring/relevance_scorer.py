"""
Relevance Scorer

검색 관련성 스코어링
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import re
import logging

logger = logging.getLogger(__name__)


class RelevanceSignal(Enum):
    """관련성 신호"""
    SEMANTIC = "semantic"       # 의미적 유사도 (벡터)
    LEXICAL = "lexical"        # 어휘적 유사도 (BM25)
    EXACT_MATCH = "exact_match"  # 정확 매칭
    ENTITY_OVERLAP = "entity_overlap"  # 엔티티 중복
    RECENCY = "recency"        # 최신성
    POPULARITY = "popularity"  # 인기도
    AUTHORITY = "authority"    # 권위성


@dataclass
class RelevanceScore:
    """관련성 점수"""
    document_id: str
    query: str
    final_score: float
    signals: Dict[str, float] = field(default_factory=dict)
    boost_applied: float = 1.0
    explanation: str = ""


@dataclass
class RelevanceScorerConfig:
    """관련성 스코어러 설정"""
    # 신호별 가중치
    semantic_weight: float = 0.4
    lexical_weight: float = 0.25
    exact_match_weight: float = 0.15
    entity_weight: float = 0.1
    recency_weight: float = 0.05
    popularity_weight: float = 0.03
    authority_weight: float = 0.02

    # BM25 파라미터
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # 부스트 설정
    title_match_boost: float = 1.5
    exact_phrase_boost: float = 2.0
    recent_doc_boost: float = 1.2
    popular_doc_boost: float = 1.1

    # 시간 감쇠
    recency_decay_days: int = 30


class RelevanceScorer:
    """검색 관련성 스코어러"""

    def __init__(self, config: Optional[RelevanceScorerConfig] = None):
        self.config = config or RelevanceScorerConfig()

    def score(
        self,
        query: str,
        document: Dict[str, Any],
        semantic_score: Optional[float] = None,
        corpus_stats: Optional[Dict[str, Any]] = None
    ) -> RelevanceScore:
        """
        검색 결과 관련성 스코어링

        Args:
            query: 검색 쿼리
            document: 문서 정보 (content, title, metadata 등)
            semantic_score: 벡터 유사도 점수
            corpus_stats: 코퍼스 통계 (IDF 계산용)

        Returns:
            관련성 점수
        """
        signals = {}

        # 1. 의미적 유사도
        if semantic_score is not None:
            signals[RelevanceSignal.SEMANTIC.value] = semantic_score
        else:
            signals[RelevanceSignal.SEMANTIC.value] = 0.0

        # 2. 어휘적 유사도 (BM25)
        content = document.get("content", "")
        lexical_score = self._calculate_bm25(query, content, corpus_stats)
        signals[RelevanceSignal.LEXICAL.value] = lexical_score

        # 3. 정확 매칭
        exact_score = self._calculate_exact_match(query, document)
        signals[RelevanceSignal.EXACT_MATCH.value] = exact_score

        # 4. 엔티티 중복
        entity_score = self._calculate_entity_overlap(query, document)
        signals[RelevanceSignal.ENTITY_OVERLAP.value] = entity_score

        # 5. 최신성
        recency_score = self._calculate_recency(document)
        signals[RelevanceSignal.RECENCY.value] = recency_score

        # 6. 인기도
        popularity_score = self._calculate_popularity(document)
        signals[RelevanceSignal.POPULARITY.value] = popularity_score

        # 7. 권위성
        authority_score = self._calculate_authority(document)
        signals[RelevanceSignal.AUTHORITY.value] = authority_score

        # 가중 합산
        weighted_score = self._combine_signals(signals)

        # 부스트 적용
        boost, boost_reasons = self._apply_boosts(query, document)

        final_score = min(weighted_score * boost, 1.0)

        # 설명 생성
        explanation = self._generate_explanation(signals, boost_reasons)

        return RelevanceScore(
            document_id=document.get("id", ""),
            query=query,
            final_score=final_score,
            signals=signals,
            boost_applied=boost,
            explanation=explanation
        )

    def score_batch(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        semantic_scores: Optional[List[float]] = None
    ) -> List[RelevanceScore]:
        """배치 스코어링"""
        # 코퍼스 통계 계산
        corpus_stats = self._compute_corpus_stats(documents)

        scores = []
        for i, doc in enumerate(documents):
            semantic = semantic_scores[i] if semantic_scores and i < len(semantic_scores) else None
            score = self.score(query, doc, semantic, corpus_stats)
            scores.append(score)

        # 점수 기준 정렬
        scores.sort(key=lambda x: x.final_score, reverse=True)

        return scores

    def _calculate_bm25(
        self,
        query: str,
        content: str,
        corpus_stats: Optional[Dict[str, Any]]
    ) -> float:
        """BM25 점수 계산"""
        if not content:
            return 0.0

        # 토큰화
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(content)

        if not query_terms or not doc_terms:
            return 0.0

        # 문서 길이
        doc_length = len(doc_terms)
        avg_doc_length = corpus_stats.get("avg_doc_length", doc_length) if corpus_stats else doc_length

        # 용어 빈도
        term_freq = {}
        for term in doc_terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        # BM25 점수 계산
        score = 0.0
        for term in query_terms:
            if term not in term_freq:
                continue

            tf = term_freq[term]

            # IDF 계산
            if corpus_stats and "doc_freq" in corpus_stats:
                df = corpus_stats["doc_freq"].get(term, 1)
                n_docs = corpus_stats.get("total_docs", 1)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            else:
                idf = 1.0

            # BM25 공식
            numerator = tf * (self.config.bm25_k1 + 1)
            denominator = tf + self.config.bm25_k1 * (
                1 - self.config.bm25_b +
                self.config.bm25_b * (doc_length / avg_doc_length)
            )

            score += idf * (numerator / denominator)

        # 정규화 (0~1)
        max_possible = len(query_terms) * 3  # 대략적인 최대값
        return min(score / max_possible, 1.0) if max_possible > 0 else 0.0

    def _calculate_exact_match(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> float:
        """정확 매칭 점수"""
        query_lower = query.lower()
        score = 0.0

        # 제목 매칭
        title = document.get("title", "").lower()
        if query_lower in title:
            score += 0.5
        elif any(term in title for term in query_lower.split()):
            score += 0.2

        # 콘텐츠 정확 매칭
        content = document.get("content", "").lower()
        if query_lower in content:
            score += 0.3

        # 메타데이터 매칭
        tags = document.get("metadata", {}).get("tags", [])
        for tag in tags:
            if query_lower in tag.lower():
                score += 0.2
                break

        return min(score, 1.0)

    def _calculate_entity_overlap(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> float:
        """엔티티 중복 점수"""
        # 쿼리에서 엔티티 추출 (간단한 방식)
        query_entities = self._extract_simple_entities(query)

        # 문서 엔티티
        doc_entities = set()
        for entity in document.get("entities", []):
            if isinstance(entity, dict):
                doc_entities.add(entity.get("name", "").lower())
            elif hasattr(entity, "name"):
                doc_entities.add(entity.name.lower())

        if not query_entities or not doc_entities:
            return 0.0

        # Jaccard 유사도
        intersection = len(query_entities & doc_entities)
        union = len(query_entities | doc_entities)

        return intersection / union if union > 0 else 0.0

    def _calculate_recency(self, document: Dict[str, Any]) -> float:
        """최신성 점수"""
        from datetime import datetime

        created_at = document.get("created_at") or document.get("metadata", {}).get("created_at")

        if not created_at:
            return 0.5

        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                return 0.5

        age_days = (datetime.now() - created_at.replace(tzinfo=None)).days

        # 지수 감쇠
        decay = math.exp(-age_days / self.config.recency_decay_days)

        return decay

    def _calculate_popularity(self, document: Dict[str, Any]) -> float:
        """인기도 점수"""
        metadata = document.get("metadata", {})

        # 조회수
        views = metadata.get("view_count", 0)
        view_score = min(math.log10(views + 1) / 4, 1.0)  # log10(10000) ≈ 4

        # 참조 수
        refs = metadata.get("reference_count", 0)
        ref_score = min(refs / 10, 1.0)

        return (view_score + ref_score) / 2

    def _calculate_authority(self, document: Dict[str, Any]) -> float:
        """권위성 점수"""
        metadata = document.get("metadata", {})
        score = 0.5  # 기본값

        # 소스 신뢰도
        source = metadata.get("source", "")
        trusted_sources = ["official", "policy", "wiki", "technical_doc"]
        if any(ts in source.lower() for ts in trusted_sources):
            score += 0.3

        # 검증 여부
        if metadata.get("verified", False):
            score += 0.2

        # 작성자 권한
        author_role = metadata.get("author_role", "")
        if author_role in ["admin", "manager", "expert"]:
            score += 0.1

        return min(score, 1.0)

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """신호 가중 합산"""
        weights = {
            RelevanceSignal.SEMANTIC.value: self.config.semantic_weight,
            RelevanceSignal.LEXICAL.value: self.config.lexical_weight,
            RelevanceSignal.EXACT_MATCH.value: self.config.exact_match_weight,
            RelevanceSignal.ENTITY_OVERLAP.value: self.config.entity_weight,
            RelevanceSignal.RECENCY.value: self.config.recency_weight,
            RelevanceSignal.POPULARITY.value: self.config.popularity_weight,
            RelevanceSignal.AUTHORITY.value: self.config.authority_weight,
        }

        total = 0.0
        for signal, value in signals.items():
            weight = weights.get(signal, 0.0)
            total += weight * value

        return total

    def _apply_boosts(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """부스트 적용"""
        boost = 1.0
        reasons = []

        query_lower = query.lower()
        title = document.get("title", "").lower()
        content = document.get("content", "").lower()

        # 제목 매칭 부스트
        if query_lower in title:
            boost *= self.config.title_match_boost
            reasons.append("title_match")

        # 정확한 구문 매칭 부스트
        if f'"{query}"' in content or query_lower in content:
            boost *= self.config.exact_phrase_boost ** 0.5  # 부분 적용
            reasons.append("phrase_match")

        # 최신 문서 부스트
        from datetime import datetime
        created_at = document.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except:
                    created_at = None

            if created_at:
                age_days = (datetime.now() - created_at.replace(tzinfo=None)).days
                if age_days < 7:
                    boost *= self.config.recent_doc_boost
                    reasons.append("recent")

        return boost, reasons

    def _generate_explanation(
        self,
        signals: Dict[str, float],
        boost_reasons: List[str]
    ) -> str:
        """점수 설명 생성"""
        # 상위 신호
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        signal_str = ", ".join(
            f"{s[0]}={s[1]:.2f}"
            for s in sorted_signals
        )

        boost_str = f", boosts=[{', '.join(boost_reasons)}]" if boost_reasons else ""

        return f"signals=[{signal_str}]{boost_str}"

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        # 간단한 토큰화 (공백 + 특수문자 분리)
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _extract_simple_entities(self, text: str) -> set:
        """간단한 엔티티 추출"""
        entities = set()

        # 대문자로 시작하는 단어
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(pn.lower() for pn in proper_nouns)

        # 한글 고유명사 패턴
        korean_nouns = re.findall(r'[가-힣]{2,}', text)
        entities.update(korean_nouns)

        return entities

    def _compute_corpus_stats(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """코퍼스 통계 계산"""
        if not documents:
            return {}

        total_length = 0
        doc_freq: Dict[str, int] = {}

        for doc in documents:
            content = doc.get("content", "")
            terms = self._tokenize(content)
            total_length += len(terms)

            unique_terms = set(terms)
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        return {
            "total_docs": len(documents),
            "avg_doc_length": total_length / len(documents) if documents else 0,
            "doc_freq": doc_freq
        }
