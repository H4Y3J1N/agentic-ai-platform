"""
Text Chunker

문서 청킹 전략 구현
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """청크 메타데이터"""
    chunk_index: int
    start_char: int
    end_char: int
    header_context: str = ""
    section_name: Optional[str] = None
    token_count: int = 0


@dataclass
class TextChunk:
    """텍스트 청크"""
    content: str
    metadata: ChunkMetadata

    @property
    def char_count(self) -> int:
        return len(self.content)


class ChunkingStrategy(str, Enum):
    """청킹 전략"""
    FIXED_SIZE = "fixed_size"           # 고정 크기
    SEMANTIC = "semantic"               # 의미 단위 (문단/섹션)
    SENTENCE = "sentence"               # 문장 단위
    RECURSIVE = "recursive"             # 재귀적 분할
    CODE_AWARE = "code_aware"           # 코드 블록 인식


@dataclass
class ChunkerConfig:
    """청커 설정"""
    chunk_size: int = 500               # 목표 청크 크기 (토큰 또는 문자)
    chunk_overlap: int = 50             # 청크 간 오버랩
    min_chunk_size: int = 100           # 최소 청크 크기
    max_chunk_size: int = 1000          # 최대 청크 크기
    use_token_count: bool = False       # True면 토큰 기준, False면 문자 기준
    respect_sentence_boundary: bool = True  # 문장 경계 존중
    preserve_code_blocks: bool = True   # 코드 블록 보존
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",     # 문단
        "\n",       # 줄바꿈
        ". ",       # 문장
        "! ",
        "? ",
        "; ",
        ", ",       # 절
        " ",        # 단어
        ""          # 문자
    ])


class Chunker(ABC):
    """청커 베이스 클래스"""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self._tokenizer: Optional[Callable[[str], int]] = None

    def set_tokenizer(self, tokenizer: Callable[[str], int]) -> None:
        """토큰 카운터 설정"""
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        if self._tokenizer:
            return self._tokenizer(text)
        # 기본: 단어 수 기반 추정 (영어 기준 ~1.3 토큰/단어)
        return int(len(text.split()) * 1.3)

    def count_length(self, text: str) -> int:
        """길이 계산 (설정에 따라 토큰 또는 문자)"""
        if self.config.use_token_count:
            return self.count_tokens(text)
        return len(text)

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """텍스트 청킹"""
        pass

    def _build_header_context(self, headers: List[str]) -> str:
        """헤더 컨텍스트 구성"""
        return " > ".join(h.strip() for h in headers if h.strip())


class FixedSizeChunker(Chunker):
    """고정 크기 청커"""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # 청크 끝 위치 계산
            end = min(start + self.config.chunk_size, len(text))

            # 문장 경계 존중
            if self.config.respect_sentence_boundary and end < len(text):
                # 문장 끝 찾기
                sentence_end = self._find_sentence_end(text, start, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        token_count=self.count_tokens(chunk_text)
                    )
                ))
                chunk_index += 1

            # 다음 시작 위치 (오버랩 고려)
            start = end - self.config.chunk_overlap

        return chunks

    def _find_sentence_end(self, text: str, start: int, end: int) -> int:
        """문장 끝 위치 찾기"""
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        best_end = end

        for ending in sentence_endings:
            pos = text.rfind(ending, start, end)
            if pos > start:
                best_end = pos + len(ending)
                break

        return best_end


class SemanticChunker(Chunker):
    """의미 단위 청커 (헤더/문단 기반)"""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        super().__init__(config)
        # 헤더 패턴 (Markdown)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        chunks = []

        # 헤더로 섹션 분리
        sections = self._split_by_headers(text)

        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(
                section["content"],
                section["headers"],
                chunk_index
            )
            for chunk in section_chunks:
                chunk.metadata.section_name = section.get("title")
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """헤더로 섹션 분리"""
        sections = []
        current_headers = []
        current_content = []
        last_end = 0

        for match in self.header_pattern.finditer(text):
            # 이전 섹션 저장
            if current_content or last_end > 0:
                content = text[last_end:match.start()].strip()
                if content:
                    sections.append({
                        "title": current_headers[-1] if current_headers else None,
                        "headers": current_headers.copy(),
                        "content": content
                    })

            # 헤더 레벨 처리
            level = len(match.group(1))
            header_text = match.group(2).strip()

            # 헤더 스택 업데이트
            while len(current_headers) >= level:
                current_headers.pop()
            current_headers.append(header_text)

            last_end = match.end()

        # 마지막 섹션
        remaining = text[last_end:].strip()
        if remaining:
            sections.append({
                "title": current_headers[-1] if current_headers else None,
                "headers": current_headers.copy(),
                "content": remaining
            })

        # 섹션이 없으면 전체를 하나로
        if not sections:
            sections.append({
                "title": None,
                "headers": [],
                "content": text
            })

        return sections

    def _chunk_section(
        self,
        content: str,
        headers: List[str],
        start_index: int
    ) -> List[TextChunk]:
        """섹션 내용 청킹"""
        header_context = self._build_header_context(headers)

        # 섹션이 충분히 짧으면 그대로 반환
        if self.count_length(content) <= self.config.max_chunk_size:
            return [TextChunk(
                content=content,
                metadata=ChunkMetadata(
                    chunk_index=start_index,
                    start_char=0,
                    end_char=len(content),
                    header_context=header_context,
                    token_count=self.count_tokens(content)
                )
            )]

        # 문단으로 분리
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = self.count_length(para)

            # 현재 청크에 추가 가능한지
            if current_length + para_length <= self.config.chunk_size:
                current_chunk.append(para)
                current_length += para_length
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            chunk_index=start_index + len(chunks),
                            start_char=chunk_start,
                            end_char=chunk_start + len(chunk_text),
                            header_context=header_context,
                            token_count=self.count_tokens(chunk_text)
                        )
                    ))
                    chunk_start += len(chunk_text) + 2

                # 새 청크 시작
                current_chunk = [para]
                current_length = para_length

        # 마지막 청크
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=start_index + len(chunks),
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    header_context=header_context,
                    token_count=self.count_tokens(chunk_text)
                )
            ))

        return chunks


class RecursiveChunker(Chunker):
    """재귀적 분할 청커"""

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        chunks = []
        self._recursive_split(text, 0, chunks, 0)
        return chunks

    def _recursive_split(
        self,
        text: str,
        start_pos: int,
        chunks: List[TextChunk],
        depth: int
    ) -> None:
        """재귀적 분할"""
        if not text.strip():
            return

        text_length = self.count_length(text)

        # 충분히 작으면 청크로 추가
        if text_length <= self.config.chunk_size:
            if text.strip():
                chunks.append(TextChunk(
                    content=text.strip(),
                    metadata=ChunkMetadata(
                        chunk_index=len(chunks),
                        start_char=start_pos,
                        end_char=start_pos + len(text),
                        token_count=self.count_tokens(text)
                    )
                ))
            return

        # 분할자 선택
        separator = self._find_best_separator(text, depth)

        if separator == "":
            # 문자 단위 분할 (최후의 수단)
            mid = len(text) // 2
            self._recursive_split(text[:mid], start_pos, chunks, depth + 1)
            self._recursive_split(text[mid:], start_pos + mid, chunks, depth + 1)
        else:
            parts = text.split(separator)
            current_pos = start_pos

            for part in parts:
                if part.strip():
                    self._recursive_split(part, current_pos, chunks, depth + 1)
                current_pos += len(part) + len(separator)

    def _find_best_separator(self, text: str, depth: int) -> str:
        """최적의 분할자 찾기"""
        for sep in self.config.separators:
            if sep in text:
                return sep
        return ""


class CodeAwareChunker(Chunker):
    """코드 블록 인식 청커"""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        super().__init__(config)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        chunks = []

        # 코드 블록 추출
        code_blocks = list(self.code_block_pattern.finditer(text))

        if not code_blocks:
            # 코드 블록 없으면 일반 청킹
            base_chunker = SemanticChunker(self.config)
            return base_chunker.chunk(text, metadata)

        # 코드와 텍스트 분리
        last_end = 0
        chunk_index = 0

        for block in code_blocks:
            # 코드 블록 전 텍스트
            pre_text = text[last_end:block.start()].strip()
            if pre_text:
                text_chunks = self._chunk_text(pre_text, chunk_index)
                chunks.extend(text_chunks)
                chunk_index += len(text_chunks)

            # 코드 블록 (분할하지 않음)
            code_content = block.group()
            if self.count_length(code_content) <= self.config.max_chunk_size:
                chunks.append(TextChunk(
                    content=code_content,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        start_char=block.start(),
                        end_char=block.end(),
                        section_name="code_block",
                        token_count=self.count_tokens(code_content)
                    )
                ))
                chunk_index += 1
            else:
                # 너무 긴 코드 블록은 분할
                code_chunks = self._chunk_code(code_content, chunk_index, block.start())
                chunks.extend(code_chunks)
                chunk_index += len(code_chunks)

            last_end = block.end()

        # 마지막 텍스트
        remaining = text[last_end:].strip()
        if remaining:
            text_chunks = self._chunk_text(remaining, chunk_index)
            chunks.extend(text_chunks)

        return chunks

    def _chunk_text(self, text: str, start_index: int) -> List[TextChunk]:
        """텍스트 청킹 (코드 제외)"""
        chunker = SemanticChunker(self.config)
        chunks = chunker.chunk(text)
        # 인덱스 조정
        for i, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = start_index + i
        return chunks

    def _chunk_code(self, code: str, start_index: int, start_char: int) -> List[TextChunk]:
        """긴 코드 블록 분할"""
        chunks = []
        lines = code.split('\n')

        current_chunk_lines = []
        current_length = 0
        chunk_start = start_char

        for line in lines:
            line_length = self.count_length(line)

            if current_length + line_length > self.config.chunk_size and current_chunk_lines:
                # 현재 청크 저장
                chunk_text = '\n'.join(current_chunk_lines)
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=start_index + len(chunks),
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        section_name="code_block",
                        token_count=self.count_tokens(chunk_text)
                    )
                ))
                chunk_start += len(chunk_text) + 1
                current_chunk_lines = []
                current_length = 0

            current_chunk_lines.append(line)
            current_length += line_length

        # 마지막 청크
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=start_index + len(chunks),
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    section_name="code_block",
                    token_count=self.count_tokens(chunk_text)
                )
            ))

        return chunks


def create_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    config: Optional[ChunkerConfig] = None
) -> Chunker:
    """청커 팩토리 함수"""
    chunkers = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.CODE_AWARE: CodeAwareChunker,
    }

    chunker_class = chunkers.get(strategy, SemanticChunker)
    return chunker_class(config)
