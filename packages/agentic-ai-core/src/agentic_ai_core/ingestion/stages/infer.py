"""
Infer Stage

메타데이터 자동 추론 스테이지
"""

from typing import Optional, List, Dict, Any
import re
import logging

from .base import Stage
from ..context import PipelineContext

logger = logging.getLogger(__name__)


class InferStage(Stage):
    """메타데이터 추론 스테이지"""

    def __init__(self, use_llm: bool = False, llm_client: Any = None):
        super().__init__("InferStage")
        self.use_llm = use_llm
        self.llm_client = llm_client

    async def process(self, context: PipelineContext) -> PipelineContext:
        """추론 실행"""
        content = context.raw_content or ""
        if not content:
            return context

        inferred = {}

        # 토픽 추출
        inferred["topics"] = self._extract_topics(content)

        # 키워드 추출
        inferred["keywords"] = self._extract_keywords(content)

        # 언어 감지
        inferred["language"] = self._detect_language(content)

        # 복잡도 점수
        inferred["complexity_score"] = self._calculate_complexity(content)

        # 제목 추론
        if not context.source_item.metadata.get("title"):
            inferred["title"] = self._infer_title(content, context)

        # 부서 추론
        department = self._infer_department(content, context.entities)
        if department:
            inferred["department"] = department

        # LLM 기반 추론 (선택적)
        if self.use_llm and self.llm_client:
            llm_inferred = await self._llm_infer(content, inferred)
            inferred.update(llm_inferred)

        context.inferred_metadata = inferred
        logger.debug(f"Inferred metadata: {list(inferred.keys())}")

        return context

    def _extract_topics(self, content: str) -> List[str]:
        """토픽 추출"""
        topics = []

        # 토픽 관련 키워드 패턴
        topic_patterns = [
            (r'#(\w+)', 'hashtag'),
            (r'(?:topic|주제|관련):\s*([^\n]+)', 'explicit'),
            (r'^#+\s+(.+)$', 'header'),
        ]

        for pattern, source in topic_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches[:5]:  # 최대 5개
                topic = match.strip()
                if topic and len(topic) > 2 and topic not in topics:
                    topics.append(topic)

        return topics[:10]  # 최대 10개

    def _extract_keywords(self, content: str) -> List[str]:
        """키워드 추출 (TF 기반)"""
        # 불용어 제거
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            '이', '그', '저', '것', '수', '등', '및', '를', '을', '에', '의',
            '가', '이', '은', '는', '다', '에서', '으로', '하다', '있다',
        }

        # 단어 추출
        words = re.findall(r'[a-zA-Z가-힣]{2,}', content.lower())

        # 빈도 계산
        freq = {}
        for word in words:
            if word not in stopwords and len(word) > 2:
                freq[word] = freq.get(word, 0) + 1

        # 상위 키워드
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:15]]

    def _detect_language(self, content: str) -> str:
        """언어 감지"""
        # 간단한 휴리스틱
        korean_chars = len(re.findall(r'[가-힣]', content))
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        japanese_chars = len(re.findall(r'[\u3040-\u30ff]', content))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))

        total = korean_chars + english_chars + japanese_chars + chinese_chars
        if total == 0:
            return "unknown"

        if korean_chars / total > 0.3:
            return "ko"
        elif japanese_chars / total > 0.3:
            return "ja"
        elif chinese_chars / total > 0.3:
            return "zh"
        else:
            return "en"

    def _calculate_complexity(self, content: str) -> float:
        """문서 복잡도 계산 (0~1)"""
        factors = []

        # 길이 기반
        length = len(content)
        length_score = min(length / 5000, 1.0)
        factors.append(length_score)

        # 문장 복잡도
        sentences = re.split(r'[.!?。]', content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_score = min(avg_sentence_length / 30, 1.0)
            factors.append(sentence_score)

        # 기술 용어 비율
        technical_patterns = [
            r'API', r'SDK', r'HTTP', r'SQL', r'JSON', r'XML',
            r'알고리즘', r'아키텍처', r'프레임워크', r'인터페이스',
        ]
        tech_count = sum(
            len(re.findall(p, content, re.IGNORECASE))
            for p in technical_patterns
        )
        tech_score = min(tech_count / 20, 1.0)
        factors.append(tech_score)

        # 코드 블록 존재
        code_blocks = len(re.findall(r'```', content))
        code_score = min(code_blocks / 6, 1.0)
        factors.append(code_score)

        return sum(factors) / len(factors) if factors else 0.5

    def _infer_title(self, content: str, context: PipelineContext) -> str:
        """제목 추론"""
        # 첫 번째 헤더
        header_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()

        # 첫 번째 줄
        first_line = content.strip().split('\n')[0]
        if first_line and len(first_line) < 100:
            return first_line.strip()

        # 소스 ID 기반
        return f"Document {context.source_item.id[:8]}"

    def _infer_department(
        self,
        content: str,
        entities: list
    ) -> Optional[str]:
        """부서 추론"""
        # 엔티티에서 부서 찾기
        from ...schema import EntityType
        for entity in entities:
            if entity.entity_type == EntityType.DEPARTMENT:
                return entity.name

        # 키워드 기반
        dept_keywords = {
            "engineering": ["개발", "engineering", "tech", "기술"],
            "hr": ["인사", "hr", "human resources", "채용"],
            "finance": ["재무", "finance", "회계", "accounting"],
            "marketing": ["마케팅", "marketing", "홍보"],
            "sales": ["영업", "sales", "세일즈"],
            "operations": ["운영", "operations", "ops"],
        }

        content_lower = content.lower()
        for dept, keywords in dept_keywords.items():
            if any(kw in content_lower for kw in keywords):
                return dept

        return None

    async def _llm_infer(
        self,
        content: str,
        partial: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM 기반 추론"""
        if not self.llm_client:
            return {}

        prompt = f"""
다음 문서를 분석하여 메타데이터를 추론하세요.

문서 내용 (일부):
{content[:2000]}

이미 추출된 정보:
{partial}

JSON 형식으로 응답 (필드: summary, target_audience, action_required):
"""
        try:
            response = await self.llm_client.complete(prompt)
            import json
            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM inference failed: {e}")
            return {}
