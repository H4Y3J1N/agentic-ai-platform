"""
Lifecycle Manager

문서 생명주기 및 해상도 관리
"""

from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """문서 생명주기 상태"""
    DRAFT = "draft"           # 작성 중
    ACTIVE = "active"         # 활성 (정상 사용)
    STALE = "stale"           # 오래됨 (업데이트 필요)
    ARCHIVED = "archived"     # 보관됨 (검색 제외)
    DELETED = "deleted"       # 삭제됨 (정리 대기)


class ResolutionLevel(Enum):
    """데이터 해상도 수준"""
    MINIMAL = "minimal"       # 메타데이터만
    STANDARD = "standard"     # 요약 + 주요 청크
    DETAILED = "detailed"     # 전체 청크
    FULL = "full"             # 전체 + 임베딩


class StorageTier(Enum):
    """저장소 계층"""
    HOT = "hot"               # 빠른 접근 (SSD)
    WARM = "warm"             # 중간 (HDD)
    COLD = "cold"             # 저비용 (Archive)
    GLACIER = "glacier"       # 장기 보관


@dataclass
class AccessPattern:
    """접근 패턴 추적"""
    document_id: str
    total_accesses: int = 0
    recent_accesses: int = 0  # 최근 30일
    last_accessed: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    access_history: List[datetime] = field(default_factory=list)

    def record_access(self) -> None:
        """접근 기록"""
        now = datetime.now()
        self.total_accesses += 1
        self.last_accessed = now
        self.access_history.append(now)

        # 최근 30일 접근 수 계산
        cutoff = now - timedelta(days=30)
        self.recent_accesses = sum(
            1 for dt in self.access_history
            if dt >= cutoff
        )

        # 히스토리 크기 제한 (최근 1000개)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]

    @property
    def access_frequency(self) -> float:
        """일 평균 접근 빈도"""
        if not self.access_history:
            return 0.0

        oldest = min(self.access_history)
        days = max((datetime.now() - oldest).days, 1)
        return self.total_accesses / days

    @property
    def is_hot(self) -> bool:
        """핫 데이터 여부"""
        return self.recent_accesses >= 10

    @property
    def is_cold(self) -> bool:
        """콜드 데이터 여부"""
        if not self.last_accessed:
            return True
        days_since_access = (datetime.now() - self.last_accessed).days
        return days_since_access > 90 and self.recent_accesses < 3


@dataclass
class LifecyclePolicy:
    """생명주기 정책"""
    # 상태 전이 기준 (일 수)
    active_to_stale_days: int = 180
    stale_to_archived_days: int = 365
    archived_to_deleted_days: int = 730

    # 해상도 조정 기준
    upgrade_access_threshold: int = 20    # 이상이면 업그레이드
    downgrade_access_threshold: int = 3   # 이하면 다운그레이드
    downgrade_days_threshold: int = 60    # 이 기간 동안 임계치 이하

    # 저장 계층 기준
    hot_access_threshold: int = 10
    cold_days_threshold: int = 90
    glacier_days_threshold: int = 365

    # 자동 삭제 설정
    auto_delete_enabled: bool = False
    min_retention_days: int = 30


@dataclass
class LifecycleTransition:
    """생명주기 전이 기록"""
    document_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    from_resolution: Optional[ResolutionLevel] = None
    to_resolution: Optional[ResolutionLevel] = None
    from_tier: Optional[StorageTier] = None
    to_tier: Optional[StorageTier] = None
    reason: str = ""
    transitioned_at: datetime = field(default_factory=datetime.now)


@dataclass
class DocumentLifecycle:
    """문서 생명주기 정보"""
    document_id: str
    state: LifecycleState = LifecycleState.ACTIVE
    resolution: ResolutionLevel = ResolutionLevel.STANDARD
    storage_tier: StorageTier = StorageTier.WARM

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    state_changed_at: datetime = field(default_factory=datetime.now)

    access_pattern: AccessPattern = field(default=None)
    transitions: List[LifecycleTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.access_pattern is None:
            self.access_pattern = AccessPattern(document_id=self.document_id)

    def record_transition(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str = ""
    ) -> None:
        """상태 전이 기록"""
        transition = LifecycleTransition(
            document_id=self.document_id,
            from_state=from_state,
            to_state=to_state,
            from_resolution=self.resolution,
            to_resolution=self.resolution,
            from_tier=self.storage_tier,
            to_tier=self.storage_tier,
            reason=reason
        )
        self.transitions.append(transition)
        self.state = to_state
        self.state_changed_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "document_id": self.document_id,
            "state": self.state.value,
            "resolution": self.resolution.value,
            "storage_tier": self.storage_tier.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state_changed_at": self.state_changed_at.isoformat(),
            "access_pattern": {
                "total_accesses": self.access_pattern.total_accesses,
                "recent_accesses": self.access_pattern.recent_accesses,
                "access_frequency": self.access_pattern.access_frequency,
            },
            "transition_count": len(self.transitions),
        }


class LifecycleManager:
    """문서 생명주기 관리자"""

    def __init__(self, policy: Optional[LifecyclePolicy] = None):
        self.policy = policy or LifecyclePolicy()
        self._lifecycles: Dict[str, DocumentLifecycle] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "on_state_change": [],
            "on_resolution_change": [],
            "on_tier_change": [],
            "on_delete": [],
        }

    def register_document(
        self,
        document_id: str,
        initial_state: LifecycleState = LifecycleState.ACTIVE,
        initial_resolution: ResolutionLevel = ResolutionLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentLifecycle:
        """문서 등록"""
        lifecycle = DocumentLifecycle(
            document_id=document_id,
            state=initial_state,
            resolution=initial_resolution,
            metadata=metadata or {}
        )
        self._lifecycles[document_id] = lifecycle
        logger.info(f"Document registered: {document_id}")
        return lifecycle

    def get_lifecycle(self, document_id: str) -> Optional[DocumentLifecycle]:
        """생명주기 정보 조회"""
        return self._lifecycles.get(document_id)

    def record_access(self, document_id: str) -> Optional[DocumentLifecycle]:
        """접근 기록"""
        lifecycle = self._lifecycles.get(document_id)
        if lifecycle:
            lifecycle.access_pattern.record_access()
            lifecycle.updated_at = datetime.now()
        return lifecycle

    def transition_state(
        self,
        document_id: str,
        new_state: LifecycleState,
        reason: str = ""
    ) -> Optional[DocumentLifecycle]:
        """상태 전이"""
        lifecycle = self._lifecycles.get(document_id)
        if not lifecycle:
            return None

        old_state = lifecycle.state

        # 유효한 전이인지 확인
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(
                f"Invalid transition: {old_state.value} -> {new_state.value}"
            )
            return None

        lifecycle.record_transition(old_state, new_state, reason)
        logger.info(
            f"State transition: {document_id} "
            f"{old_state.value} -> {new_state.value}"
        )

        # 훅 실행
        self._execute_hooks("on_state_change", lifecycle, old_state, new_state)

        return lifecycle

    def upgrade_resolution(
        self,
        document_id: str,
        target_resolution: Optional[ResolutionLevel] = None
    ) -> Optional[DocumentLifecycle]:
        """해상도 업그레이드"""
        lifecycle = self._lifecycles.get(document_id)
        if not lifecycle:
            return None

        old_resolution = lifecycle.resolution

        if target_resolution:
            new_resolution = target_resolution
        else:
            # 자동 결정
            resolution_order = [
                ResolutionLevel.MINIMAL,
                ResolutionLevel.STANDARD,
                ResolutionLevel.DETAILED,
                ResolutionLevel.FULL
            ]
            current_idx = resolution_order.index(old_resolution)
            if current_idx < len(resolution_order) - 1:
                new_resolution = resolution_order[current_idx + 1]
            else:
                return lifecycle  # 이미 최고 해상도

        lifecycle.resolution = new_resolution
        lifecycle.updated_at = datetime.now()

        logger.info(
            f"Resolution upgraded: {document_id} "
            f"{old_resolution.value} -> {new_resolution.value}"
        )

        self._execute_hooks(
            "on_resolution_change",
            lifecycle, old_resolution, new_resolution
        )

        return lifecycle

    def downgrade_resolution(
        self,
        document_id: str,
        target_resolution: Optional[ResolutionLevel] = None
    ) -> Optional[DocumentLifecycle]:
        """해상도 다운그레이드"""
        lifecycle = self._lifecycles.get(document_id)
        if not lifecycle:
            return None

        old_resolution = lifecycle.resolution

        if target_resolution:
            new_resolution = target_resolution
        else:
            resolution_order = [
                ResolutionLevel.MINIMAL,
                ResolutionLevel.STANDARD,
                ResolutionLevel.DETAILED,
                ResolutionLevel.FULL
            ]
            current_idx = resolution_order.index(old_resolution)
            if current_idx > 0:
                new_resolution = resolution_order[current_idx - 1]
            else:
                return lifecycle  # 이미 최저 해상도

        lifecycle.resolution = new_resolution
        lifecycle.updated_at = datetime.now()

        logger.info(
            f"Resolution downgraded: {document_id} "
            f"{old_resolution.value} -> {new_resolution.value}"
        )

        self._execute_hooks(
            "on_resolution_change",
            lifecycle, old_resolution, new_resolution
        )

        return lifecycle

    def update_storage_tier(
        self,
        document_id: str,
        new_tier: StorageTier
    ) -> Optional[DocumentLifecycle]:
        """저장소 계층 변경"""
        lifecycle = self._lifecycles.get(document_id)
        if not lifecycle:
            return None

        old_tier = lifecycle.storage_tier
        lifecycle.storage_tier = new_tier
        lifecycle.updated_at = datetime.now()

        logger.info(
            f"Storage tier changed: {document_id} "
            f"{old_tier.value} -> {new_tier.value}"
        )

        self._execute_hooks("on_tier_change", lifecycle, old_tier, new_tier)

        return lifecycle

    def evaluate_and_apply_policy(
        self,
        document_id: str
    ) -> Optional[DocumentLifecycle]:
        """정책 평가 및 적용"""
        lifecycle = self._lifecycles.get(document_id)
        if not lifecycle:
            return None

        now = datetime.now()
        access = lifecycle.access_pattern

        # 1. 상태 전이 평가
        if lifecycle.state == LifecycleState.ACTIVE:
            days_since_update = (now - lifecycle.updated_at).days
            if days_since_update > self.policy.active_to_stale_days:
                self.transition_state(
                    document_id,
                    LifecycleState.STALE,
                    f"No updates for {days_since_update} days"
                )

        elif lifecycle.state == LifecycleState.STALE:
            days_since_state_change = (now - lifecycle.state_changed_at).days
            if days_since_state_change > self.policy.stale_to_archived_days:
                self.transition_state(
                    document_id,
                    LifecycleState.ARCHIVED,
                    f"Stale for {days_since_state_change} days"
                )

        elif lifecycle.state == LifecycleState.ARCHIVED:
            if self.policy.auto_delete_enabled:
                days_archived = (now - lifecycle.state_changed_at).days
                if days_archived > self.policy.archived_to_deleted_days:
                    self.transition_state(
                        document_id,
                        LifecycleState.DELETED,
                        f"Archived for {days_archived} days"
                    )

        # 2. 해상도 조정 평가
        if access.recent_accesses >= self.policy.upgrade_access_threshold:
            if lifecycle.resolution != ResolutionLevel.FULL:
                self.upgrade_resolution(document_id)

        elif access.recent_accesses <= self.policy.downgrade_access_threshold:
            if access.last_accessed:
                days_since_access = (now - access.last_accessed).days
                if days_since_access >= self.policy.downgrade_days_threshold:
                    if lifecycle.resolution != ResolutionLevel.MINIMAL:
                        self.downgrade_resolution(document_id)

        # 3. 저장소 계층 평가
        if access.is_hot:
            if lifecycle.storage_tier != StorageTier.HOT:
                self.update_storage_tier(document_id, StorageTier.HOT)

        elif access.is_cold:
            if lifecycle.storage_tier not in [StorageTier.COLD, StorageTier.GLACIER]:
                self.update_storage_tier(document_id, StorageTier.COLD)

        return lifecycle

    def get_documents_by_state(
        self,
        state: LifecycleState
    ) -> List[DocumentLifecycle]:
        """상태별 문서 조회"""
        return [
            lc for lc in self._lifecycles.values()
            if lc.state == state
        ]

    def get_documents_by_tier(
        self,
        tier: StorageTier
    ) -> List[DocumentLifecycle]:
        """저장소 계층별 문서 조회"""
        return [
            lc for lc in self._lifecycles.values()
            if lc.storage_tier == tier
        ]

    def get_candidates_for_cleanup(self) -> List[DocumentLifecycle]:
        """정리 대상 문서 조회"""
        candidates = []

        for lifecycle in self._lifecycles.values():
            # 삭제 상태 문서
            if lifecycle.state == LifecycleState.DELETED:
                candidates.append(lifecycle)
                continue

            # 오래된 아카이브 문서
            if lifecycle.state == LifecycleState.ARCHIVED:
                days = (datetime.now() - lifecycle.state_changed_at).days
                if days > self.policy.archived_to_deleted_days:
                    candidates.append(lifecycle)

        return candidates

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보"""
        total = len(self._lifecycles)

        state_counts = {}
        for state in LifecycleState:
            state_counts[state.value] = sum(
                1 for lc in self._lifecycles.values()
                if lc.state == state
            )

        resolution_counts = {}
        for res in ResolutionLevel:
            resolution_counts[res.value] = sum(
                1 for lc in self._lifecycles.values()
                if lc.resolution == res
            )

        tier_counts = {}
        for tier in StorageTier:
            tier_counts[tier.value] = sum(
                1 for lc in self._lifecycles.values()
                if lc.storage_tier == tier
            )

        return {
            "total_documents": total,
            "by_state": state_counts,
            "by_resolution": resolution_counts,
            "by_tier": tier_counts,
        }

    def add_hook(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """이벤트 훅 추가"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _execute_hooks(self, event: str, *args) -> None:
        """훅 실행"""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Hook execution error: {e}")

    def _is_valid_transition(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState
    ) -> bool:
        """유효한 상태 전이인지 확인"""
        valid_transitions = {
            LifecycleState.DRAFT: {
                LifecycleState.ACTIVE,
                LifecycleState.DELETED
            },
            LifecycleState.ACTIVE: {
                LifecycleState.STALE,
                LifecycleState.ARCHIVED,
                LifecycleState.DELETED
            },
            LifecycleState.STALE: {
                LifecycleState.ACTIVE,
                LifecycleState.ARCHIVED,
                LifecycleState.DELETED
            },
            LifecycleState.ARCHIVED: {
                LifecycleState.ACTIVE,
                LifecycleState.DELETED
            },
            LifecycleState.DELETED: set(),  # 삭제 상태에서는 전이 불가
        }

        return to_state in valid_transitions.get(from_state, set())


class BatchLifecycleManager:
    """배치 생명주기 관리자"""

    def __init__(self, manager: Optional[LifecycleManager] = None):
        self.manager = manager or LifecycleManager()

    def evaluate_all(self) -> Dict[str, Any]:
        """전체 문서 정책 평가"""
        results = {
            "evaluated": 0,
            "state_changes": 0,
            "resolution_changes": 0,
            "tier_changes": 0,
        }

        for doc_id in list(self.manager._lifecycles.keys()):
            before = self.manager.get_lifecycle(doc_id)
            if not before:
                continue

            old_state = before.state
            old_resolution = before.resolution
            old_tier = before.storage_tier

            self.manager.evaluate_and_apply_policy(doc_id)

            after = self.manager.get_lifecycle(doc_id)
            if after:
                results["evaluated"] += 1
                if after.state != old_state:
                    results["state_changes"] += 1
                if after.resolution != old_resolution:
                    results["resolution_changes"] += 1
                if after.storage_tier != old_tier:
                    results["tier_changes"] += 1

        return results

    def cleanup_deleted(self) -> List[str]:
        """삭제된 문서 정리"""
        deleted_ids = []

        candidates = self.manager.get_candidates_for_cleanup()
        for lifecycle in candidates:
            if lifecycle.state == LifecycleState.DELETED:
                # 훅 실행
                self.manager._execute_hooks("on_delete", lifecycle)
                deleted_ids.append(lifecycle.document_id)

        # 실제 삭제
        for doc_id in deleted_ids:
            del self.manager._lifecycles[doc_id]

        logger.info(f"Cleaned up {len(deleted_ids)} deleted documents")
        return deleted_ids

    def migrate_tier(
        self,
        from_tier: StorageTier,
        to_tier: StorageTier,
        limit: Optional[int] = None
    ) -> List[str]:
        """저장소 계층 마이그레이션"""
        migrated = []

        documents = self.manager.get_documents_by_tier(from_tier)
        if limit:
            documents = documents[:limit]

        for lifecycle in documents:
            self.manager.update_storage_tier(lifecycle.document_id, to_tier)
            migrated.append(lifecycle.document_id)

        logger.info(
            f"Migrated {len(migrated)} documents "
            f"from {from_tier.value} to {to_tier.value}"
        )
        return migrated
