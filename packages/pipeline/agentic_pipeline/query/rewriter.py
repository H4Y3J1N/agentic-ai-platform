"""
Query Rewriter

쿼리 재작성 및 확장
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class RewriteStrategy(Enum):
    """재작성 전략"""
    EXPANSION = "expansion"      # 동의어/관련어 확장
    DECOMPOSITION = "decomposition"  # 복합 쿼리 분해
    REFINEMENT = "refinement"    # 쿼리 정제
    TRANSLATION = "translation"  # 언어 변환


@dataclass
class RewrittenQuery:
    """재작성된 쿼리"""
    original: str
    rewritten: str
    strategy: RewriteStrategy
    expansions: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    intent: str
    entities: List[str]
    keywords: List[str]
    temporal_references: List[str]
    negations: List[str]
    is_question: bool
    language: str
    complexity: str  # simple, moderate, complex


class QueryRewriter:
    """쿼리 재작성기"""

    def __init__(
        self,
        synonyms: Optional[Dict[str, List[str]]] = None,
        stopwords: Optional[Set[str]] = None,
        domain_terms: Optional[Dict[str, str]] = None
    ):
        self.synonyms = synonyms or self._default_synonyms()
        self.stopwords = stopwords or self._default_stopwords()
        self.domain_terms = domain_terms or {}

    def analyze(self, query: str) -> QueryAnalysis:
        """쿼리 분석"""
        query_lower = query.lower()

        # 의도 파악
        intent = self._detect_intent(query_lower)

        # 엔티티 추출
        entities = self._extract_entities(query)

        # 키워드 추출
        keywords = self._extract_keywords(query)

        # 시간 참조 추출
        temporal = self._extract_temporal(query_lower)

        # 부정 추출
        negations = self._extract_negations(query_lower)

        # 질문 여부
        is_question = self._is_question(query)

        # 언어 감지
        language = self._detect_language(query)

        # 복잡도 평가
        complexity = self._assess_complexity(query, entities, keywords)

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            keywords=keywords,
            temporal_references=temporal,
            negations=negations,
            is_question=is_question,
            language=language,
            complexity=complexity
        )

    def rewrite(
        self,
        query: str,
        strategies: Optional[List[RewriteStrategy]] = None
    ) -> List[RewrittenQuery]:
        """쿼리 재작성"""
        strategies = strategies or [
            RewriteStrategy.EXPANSION,
            RewriteStrategy.REFINEMENT
        ]

        results = []

        for strategy in strategies:
            if strategy == RewriteStrategy.EXPANSION:
                result = self._expand_query(query)
            elif strategy == RewriteStrategy.DECOMPOSITION:
                result = self._decompose_query(query)
            elif strategy == RewriteStrategy.REFINEMENT:
                result = self._refine_query(query)
            elif strategy == RewriteStrategy.TRANSLATION:
                result = self._translate_query(query)
            else:
                continue

            if result:
                results.append(result)

        return results

    def expand(self, query: str) -> RewrittenQuery:
        """동의어/관련어 확장"""
        return self._expand_query(query)

    def decompose(self, query: str) -> RewrittenQuery:
        """복합 쿼리 분해"""
        return self._decompose_query(query)

    def refine(self, query: str) -> RewrittenQuery:
        """쿼리 정제"""
        return self._refine_query(query)

    # ==================
    # Intent Detection
    # ==================

    def _detect_intent(self, query: str) -> str:
        """의도 감지"""
        # 정책/규정 관련
        if any(kw in query for kw in ["정책", "규정", "policy", "규칙", "rule"]):
            return "policy_inquiry"

        # 절차/방법 관련
        if any(kw in query for kw in ["어떻게", "방법", "how", "절차", "process"]):
            return "procedure_inquiry"

        # 정의/설명 관련
        if any(kw in query for kw in ["무엇", "what", "뭐", "정의", "설명"]):
            return "definition_inquiry"

        # 위치/장소 관련
        if any(kw in query for kw in ["어디", "where", "위치", "장소"]):
            return "location_inquiry"

        # 시간 관련
        if any(kw in query for kw in ["언제", "when", "기한", "마감"]):
            return "temporal_inquiry"

        # 사람 관련
        if any(kw in query for kw in ["누구", "who", "담당자", "책임자"]):
            return "person_inquiry"

        # 비교 관련
        if any(kw in query for kw in ["차이", "비교", "versus", "vs", "다른"]):
            return "comparison"

        # 목록 관련
        if any(kw in query for kw in ["목록", "list", "종류", "전체"]):
            return "listing"

        return "general_inquiry"

    # ==================
    # Entity Extraction
    # ==================

    def _extract_entities(self, query: str) -> List[str]:
        """엔티티 추출"""
        entities = []

        # 따옴표 안의 내용
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted)

        # 고유명사 패턴 (대문자로 시작)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        entities.extend(proper_nouns)

        # 한글 고유명사 패턴 (조사 앞의 명사)
        korean_nouns = re.findall(r'([가-힣]+)(?:은|는|이|가|을|를|의|에서|에게|로|으로)', query)
        entities.extend(korean_nouns)

        # 도메인 용어
        for term in self.domain_terms:
            if term.lower() in query.lower():
                entities.append(term)

        return list(set(entities))

    # ==================
    # Keyword Extraction
    # ==================

    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 토큰화
        tokens = re.findall(r'\b\w+\b', query.lower())

        # 불용어 제거
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        return keywords

    # ==================
    # Temporal Extraction
    # ==================

    def _extract_temporal(self, query: str) -> List[str]:
        """시간 참조 추출"""
        temporal = []

        # 날짜 패턴
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}월\s*\d{1,2}일',
            r'(?:오늘|내일|어제|모레)',
            r'(?:이번|다음|지난)\s*(?:주|달|월|년)',
            r'\d+\s*(?:일|주|달|월|년)\s*(?:전|후|내)',
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            temporal.extend(matches)

        return temporal

    # ==================
    # Negation Extraction
    # ==================

    def _extract_negations(self, query: str) -> List[str]:
        """부정 표현 추출"""
        negation_patterns = [
            r'(?:않|안|못|없|아닌)[가-힣]*',
            r'\bnot\b',
            r'\bno\b',
            r'\bnever\b',
            r'\bwithout\b',
            r'제외',
            r'빼고',
        ]

        negations = []
        for pattern in negation_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            negations.extend(matches)

        return negations

    # ==================
    # Query Classification
    # ==================

    def _is_question(self, query: str) -> bool:
        """질문 여부 판단"""
        question_markers = [
            r'\?$',
            r'^(?:what|who|where|when|why|how|which)',
            r'(?:뭐|무엇|누구|어디|언제|왜|어떻게|몇)',
            r'(?:인가요|인가|일까|일까요|나요|는지|인지)',
        ]

        for pattern in question_markers:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def _detect_language(self, query: str) -> str:
        """언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))

        if korean_chars > english_chars:
            return "ko"
        elif english_chars > korean_chars:
            return "en"
        else:
            return "mixed"

    def _assess_complexity(
        self,
        query: str,
        entities: List[str],
        keywords: List[str]
    ) -> str:
        """복잡도 평가"""
        # 복합 조건
        has_conjunction = bool(re.search(r'\b(?:and|or|but|그리고|또는|하지만)\b', query, re.IGNORECASE))
        has_comparison = bool(re.search(r'\b(?:than|versus|vs|보다|비교)\b', query, re.IGNORECASE))
        has_temporal = bool(self._extract_temporal(query.lower()))

        score = 0
        score += len(entities) * 2
        score += len(keywords)
        score += 3 if has_conjunction else 0
        score += 3 if has_comparison else 0
        score += 2 if has_temporal else 0

        if score <= 5:
            return "simple"
        elif score <= 12:
            return "moderate"
        else:
            return "complex"

    # ==================
    # Query Expansion
    # ==================

    def _expand_query(self, query: str) -> RewrittenQuery:
        """쿼리 확장"""
        tokens = re.findall(r'\b\w+\b', query.lower())
        expansions = []

        for token in tokens:
            if token in self.synonyms:
                expansions.extend(self.synonyms[token])

        # 확장된 키워드를 쿼리에 추가
        expanded_parts = [query]
        if expansions:
            expanded_parts.append(" OR ".join(set(expansions[:5])))

        rewritten = " ".join(expanded_parts)

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            strategy=RewriteStrategy.EXPANSION,
            expansions=list(set(expansions)),
            metadata={"expansion_count": len(expansions)}
        )

    # ==================
    # Query Decomposition
    # ==================

    def _decompose_query(self, query: str) -> RewrittenQuery:
        """복합 쿼리 분해"""
        sub_queries = []

        # AND/그리고로 분리
        parts = re.split(r'\s+(?:and|그리고)\s+', query, flags=re.IGNORECASE)
        if len(parts) > 1:
            sub_queries.extend(parts)

        # OR/또는로 분리
        if not sub_queries:
            parts = re.split(r'\s+(?:or|또는)\s+', query, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_queries.extend(parts)

        # 분리되지 않으면 원본 유지
        if not sub_queries:
            sub_queries = [query]

        return RewrittenQuery(
            original=query,
            rewritten=query,
            strategy=RewriteStrategy.DECOMPOSITION,
            sub_queries=[sq.strip() for sq in sub_queries],
            metadata={"sub_query_count": len(sub_queries)}
        )

    # ==================
    # Query Refinement
    # ==================

    def _refine_query(self, query: str) -> RewrittenQuery:
        """쿼리 정제"""
        refined = query

        # 불용어 제거
        tokens = refined.split()
        refined_tokens = [t for t in tokens if t.lower() not in self.stopwords]

        # 특수문자 정리
        refined = " ".join(refined_tokens)
        refined = re.sub(r'[^\w\s가-힣]', ' ', refined)
        refined = re.sub(r'\s+', ' ', refined).strip()

        return RewrittenQuery(
            original=query,
            rewritten=refined,
            strategy=RewriteStrategy.REFINEMENT,
            metadata={"removed_tokens": len(tokens) - len(refined_tokens)}
        )

    # ==================
    # Query Translation
    # ==================

    def _translate_query(self, query: str) -> RewrittenQuery:
        """쿼리 번역 (간단한 용어 매핑)"""
        translated = query

        # 도메인 용어 번역
        for term, translation in self.domain_terms.items():
            translated = re.sub(
                rf'\b{re.escape(term)}\b',
                translation,
                translated,
                flags=re.IGNORECASE
            )

        return RewrittenQuery(
            original=query,
            rewritten=translated,
            strategy=RewriteStrategy.TRANSLATION,
            metadata={"translations_applied": query != translated}
        )

    # ==================
    # Default Data
    # ==================

    def _default_synonyms(self) -> Dict[str, List[str]]:
        """기본 동의어 사전"""
        return {
            # 한글
            "휴가": ["연차", "휴일", "쉬는날", "leave", "vacation"],
            "정책": ["규정", "규칙", "가이드라인", "policy", "rule"],
            "회의": ["미팅", "meeting", "모임", "회의록"],
            "프로젝트": ["project", "사업", "과제"],
            "담당자": ["책임자", "담당", "manager", "owner"],
            "승인": ["결재", "approval", "허가"],
            "신청": ["요청", "request", "apply"],
            # 영어
            "policy": ["rule", "regulation", "guideline", "정책"],
            "meeting": ["conference", "회의", "미팅"],
            "project": ["initiative", "프로젝트", "사업"],
            "employee": ["staff", "worker", "직원"],
            "manager": ["supervisor", "담당자", "매니저"],
        }

    def _default_stopwords(self) -> Set[str]:
        """기본 불용어"""
        return {
            # 한글
            "은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로",
            "와", "과", "도", "만", "뿐", "하다", "되다", "있다", "없다",
            "것", "수", "등", "및", "또는", "그리고", "하지만", "그러나",
            # 영어
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "as", "until", "while", "this", "that", "these", "those",
        }
