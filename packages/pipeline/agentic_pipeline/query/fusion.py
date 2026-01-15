"""
Result Fusion

다중 소스 검색 결과 융합
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """융합 전략"""
    RRF = "rrf"             # Reciprocal Rank Fusion
    WEIGHTED = "weighted"   # 가중 합산
    UNION = "union"         # 합집합
    INTERSECTION = "intersection"  # 교집합
    CASCADE = "cascade"     # 단계적 필터링
    BORDA = "borda"         # Borda Count


@dataclass
class SearchResultItem:
    """검색 결과 아이템"""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


@dataclass
class FusionResult:
    """융합 결과"""
    items: List[SearchResultItem]
    strategy_used: FusionStrategy
    source_contributions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultFusion:
    """결과 융합기"""

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF의 k 파라미터 (기본값 60)
        """
        self.k = k

    def fuse(
        self,
        results: List[SearchResultItem],
        strategy: FusionStrategy = FusionStrategy.RRF,
        weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResultItem]:
        """
        결과 융합

        Args:
            results: 검색 결과 리스트
            strategy: 융합 전략
            weights: 소스별 가중치

        Returns:
            융합된 결과 리스트
        """
        weights = weights or {}

        if strategy == FusionStrategy.RRF:
            return self._rrf_fusion(results)
        elif strategy == FusionStrategy.WEIGHTED:
            return self._weighted_fusion(results, weights)
        elif strategy == FusionStrategy.UNION:
            return self._union_fusion(results)
        elif strategy == FusionStrategy.INTERSECTION:
            return self._intersection_fusion(results)
        elif strategy == FusionStrategy.CASCADE:
            return self._cascade_fusion(results, weights)
        elif strategy == FusionStrategy.BORDA:
            return self._borda_fusion(results)
        else:
            return self._rrf_fusion(results)

    def _rrf_fusion(
        self,
        results: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """
        Reciprocal Rank Fusion

        RRF Score = Σ 1 / (k + rank_i)

        각 소스에서의 순위를 기반으로 점수 계산
        """
        # 소스별로 결과 그룹화 및 순위 부여
        source_results: Dict[str, List[SearchResultItem]] = defaultdict(list)
        for item in results:
            source_results[item.source].append(item)

        # 각 소스 내에서 점수로 정렬 후 순위 부여
        for source, items in source_results.items():
            items.sort(key=lambda x: x.score, reverse=True)
            for rank, item in enumerate(items, 1):
                item.rank = rank

        # RRF 점수 계산
        rrf_scores: Dict[str, float] = defaultdict(float)
        id_to_item: Dict[str, SearchResultItem] = {}

        for item in results:
            rrf_scores[item.id] += 1.0 / (self.k + item.rank)
            if item.id not in id_to_item:
                id_to_item[item.id] = item

        # 최종 결과 구성
        fused_results = []
        for item_id, rrf_score in rrf_scores.items():
            item = id_to_item[item_id]
            fused_item = SearchResultItem(
                id=item.id,
                content=item.content,
                score=rrf_score,
                source=item.source,
                metadata={
                    **item.metadata,
                    "fusion_method": "rrf",
                    "original_score": item.score
                }
            )
            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _weighted_fusion(
        self,
        results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """
        가중 합산 융합

        Final Score = Σ weight_source × score_source
        """
        # 기본 가중치
        default_weight = 1.0 / max(len(set(r.source for r in results)), 1)

        # ID별 가중 점수 합산
        weighted_scores: Dict[str, float] = defaultdict(float)
        score_counts: Dict[str, int] = defaultdict(int)
        id_to_item: Dict[str, SearchResultItem] = {}

        for item in results:
            weight = weights.get(item.source, weights.get(
                item.metadata.get("step_id", ""),
                default_weight
            ))
            weighted_scores[item.id] += weight * item.score
            score_counts[item.id] += 1

            if item.id not in id_to_item:
                id_to_item[item.id] = item

        # 최종 결과 구성
        fused_results = []
        for item_id, total_score in weighted_scores.items():
            item = id_to_item[item_id]
            # 평균 가중 점수
            avg_score = total_score / score_counts[item_id]

            fused_item = SearchResultItem(
                id=item.id,
                content=item.content,
                score=avg_score,
                source=item.source,
                metadata={
                    **item.metadata,
                    "fusion_method": "weighted",
                    "original_score": item.score,
                    "appearance_count": score_counts[item_id]
                }
            )
            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _union_fusion(
        self,
        results: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """
        합집합 융합

        모든 소스의 결과를 합치고 중복 제거
        """
        seen_ids: Dict[str, SearchResultItem] = {}

        for item in results:
            if item.id not in seen_ids:
                seen_ids[item.id] = item
            else:
                # 더 높은 점수 유지
                if item.score > seen_ids[item.id].score:
                    seen_ids[item.id] = item

        fused_results = list(seen_ids.values())
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # 메타데이터 추가
        for item in fused_results:
            item.metadata["fusion_method"] = "union"

        return fused_results

    def _intersection_fusion(
        self,
        results: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """
        교집합 융합

        모든 소스에 존재하는 결과만 반환
        """
        # 소스별 ID 집합
        source_ids: Dict[str, set] = defaultdict(set)
        id_to_items: Dict[str, List[SearchResultItem]] = defaultdict(list)

        for item in results:
            source_ids[item.source].add(item.id)
            id_to_items[item.id].append(item)

        # 교집합 계산
        if not source_ids:
            return []

        common_ids = set.intersection(*source_ids.values()) if source_ids else set()

        # 교집합 결과 구성
        fused_results = []
        for item_id in common_ids:
            items = id_to_items[item_id]
            # 평균 점수 사용
            avg_score = sum(i.score for i in items) / len(items)

            fused_item = SearchResultItem(
                id=item_id,
                content=items[0].content,
                score=avg_score,
                source="intersection",
                metadata={
                    **items[0].metadata,
                    "fusion_method": "intersection",
                    "source_count": len(items)
                }
            )
            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _cascade_fusion(
        self,
        results: List[SearchResultItem],
        weights: Dict[str, float]
    ) -> List[SearchResultItem]:
        """
        단계적 필터링 융합

        가중치 순으로 소스를 처리하며 점진적 필터링
        """
        # 가중치 기준 소스 정렬
        source_priority = sorted(
            set(r.source for r in results),
            key=lambda s: weights.get(s, 0.5),
            reverse=True
        )

        # 단계적 처리
        candidate_ids: Optional[set] = None
        id_to_items: Dict[str, List[SearchResultItem]] = defaultdict(list)

        for source in source_priority:
            source_items = [r for r in results if r.source == source]
            source_item_ids = {item.id for item in source_items}

            for item in source_items:
                id_to_items[item.id].append(item)

            if candidate_ids is None:
                candidate_ids = source_item_ids
            else:
                # 교집합 또는 일정 비율 유지
                intersection = candidate_ids & source_item_ids
                if len(intersection) >= len(candidate_ids) * 0.3:
                    candidate_ids = intersection
                # 너무 적으면 합집합
                elif len(intersection) < 5:
                    candidate_ids = candidate_ids | source_item_ids

        # 최종 결과 구성
        fused_results = []
        for item_id in (candidate_ids or set()):
            items = id_to_items.get(item_id, [])
            if not items:
                continue

            # 가중 평균 점수
            total_weight = 0
            weighted_score = 0
            for item in items:
                w = weights.get(item.source, 0.5)
                weighted_score += w * item.score
                total_weight += w

            final_score = weighted_score / total_weight if total_weight > 0 else 0

            fused_item = SearchResultItem(
                id=item_id,
                content=items[0].content,
                score=final_score,
                source=items[0].source,
                metadata={
                    **items[0].metadata,
                    "fusion_method": "cascade",
                    "source_count": len(items)
                }
            )
            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _borda_fusion(
        self,
        results: List[SearchResultItem]
    ) -> List[SearchResultItem]:
        """
        Borda Count 융합

        각 소스에서의 순위를 점수로 변환하여 합산
        """
        # 소스별 결과 그룹화
        source_results: Dict[str, List[SearchResultItem]] = defaultdict(list)
        for item in results:
            source_results[item.source].append(item)

        # Borda 점수 계산
        borda_scores: Dict[str, float] = defaultdict(float)
        id_to_item: Dict[str, SearchResultItem] = {}

        for source, items in source_results.items():
            # 점수 기준 정렬
            items.sort(key=lambda x: x.score, reverse=True)
            n = len(items)

            for rank, item in enumerate(items):
                # Borda 점수: n - rank (1등이 가장 높음)
                borda_scores[item.id] += n - rank

                if item.id not in id_to_item:
                    id_to_item[item.id] = item

        # 최종 결과 구성
        fused_results = []
        for item_id, borda_score in borda_scores.items():
            item = id_to_item[item_id]

            fused_item = SearchResultItem(
                id=item.id,
                content=item.content,
                score=borda_score,
                source=item.source,
                metadata={
                    **item.metadata,
                    "fusion_method": "borda",
                    "original_score": item.score
                }
            )
            fused_results.append(fused_item)

        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results


class DiversityReranker:
    """다양성 기반 재정렬"""

    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: 관련성 vs 다양성 균형 (0=다양성, 1=관련성)
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        results: List[SearchResultItem],
        top_k: int = 10
    ) -> List[SearchResultItem]:
        """
        MMR(Maximal Marginal Relevance) 기반 재정렬

        다양성을 고려하여 결과 재정렬
        """
        if len(results) <= top_k:
            return results

        selected: List[SearchResultItem] = []
        remaining = results.copy()

        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # 관련성 점수
                relevance = candidate.score

                # 다양성 점수 (선택된 것들과의 최대 유사도)
                if selected:
                    max_similarity = max(
                        self._compute_similarity(candidate, s)
                        for s in selected
                    )
                else:
                    max_similarity = 0

                # MMR 점수
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _compute_similarity(
        self,
        item1: SearchResultItem,
        item2: SearchResultItem
    ) -> float:
        """두 결과 간 유사도 계산"""
        # 소스 기반 유사도
        source_sim = 1.0 if item1.source == item2.source else 0.0

        # 메타데이터 기반 유사도
        meta_sim = 0.0
        if item1.metadata and item2.metadata:
            common_keys = set(item1.metadata.keys()) & set(item2.metadata.keys())
            if common_keys:
                matches = sum(
                    1 for k in common_keys
                    if item1.metadata[k] == item2.metadata[k]
                )
                meta_sim = matches / len(common_keys)

        # 콘텐츠 기반 유사도 (단어 중복)
        words1 = set(item1.content.lower().split())
        words2 = set(item2.content.lower().split())
        if words1 and words2:
            content_sim = len(words1 & words2) / len(words1 | words2)
        else:
            content_sim = 0.0

        # 가중 평균
        return 0.3 * source_sim + 0.2 * meta_sim + 0.5 * content_sim
