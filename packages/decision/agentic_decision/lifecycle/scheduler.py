"""
Lifecycle Scheduler

정리 작업 및 유지보수 스케줄러
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod

from .manager import (
    LifecycleManager,
    BatchLifecycleManager,
    LifecycleState,
    ResolutionLevel,
    StorageTier,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduleType(Enum):
    """스케줄 타입"""
    ONCE = "once"             # 일회성
    INTERVAL = "interval"     # 주기적 (초 단위)
    DAILY = "daily"           # 매일 특정 시간
    WEEKLY = "weekly"         # 매주 특정 요일/시간
    CRON = "cron"             # Cron 표현식


@dataclass
class TaskResult:
    """작업 실행 결과"""
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """스케줄된 작업"""
    task_id: str
    name: str
    schedule_type: ScheduleType
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True

    # 스케줄 설정
    interval_seconds: Optional[int] = None
    daily_time: Optional[str] = None  # "HH:MM"
    weekly_day: Optional[int] = None  # 0=월, 6=일
    cron_expression: Optional[str] = None

    # 실행 정보
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0

    # 설정
    max_retries: int = 3
    timeout_seconds: int = 300
    max_consecutive_failures: int = 5

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_next_run(self) -> Optional[datetime]:
        """다음 실행 시간 계산"""
        now = datetime.now()

        if self.schedule_type == ScheduleType.ONCE:
            if self.run_count == 0:
                return now
            return None

        elif self.schedule_type == ScheduleType.INTERVAL:
            if self.interval_seconds:
                base = self.last_run or now
                return base + timedelta(seconds=self.interval_seconds)

        elif self.schedule_type == ScheduleType.DAILY:
            if self.daily_time:
                hour, minute = map(int, self.daily_time.split(":"))
                next_time = now.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                if next_time <= now:
                    next_time += timedelta(days=1)
                return next_time

        elif self.schedule_type == ScheduleType.WEEKLY:
            if self.weekly_day is not None and self.daily_time:
                hour, minute = map(int, self.daily_time.split(":"))
                days_ahead = self.weekly_day - now.weekday()
                if days_ahead < 0:
                    days_ahead += 7
                next_time = now.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                next_time += timedelta(days=days_ahead)
                if next_time <= now:
                    next_time += timedelta(days=7)
                return next_time

        return None

    def should_run(self) -> bool:
        """실행 여부 확인"""
        if not self.enabled:
            return False

        if self.consecutive_failures >= self.max_consecutive_failures:
            return False

        if self.next_run and datetime.now() >= self.next_run:
            return True

        return False


class MaintenanceTask(ABC):
    """유지보수 작업 베이스 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """작업 이름"""
        pass

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """작업 실행"""
        pass

    def on_success(self, result: Dict[str, Any]) -> None:
        """성공 콜백"""
        pass

    def on_failure(self, error: Exception) -> None:
        """실패 콜백"""
        pass


class PolicyEvaluationTask(MaintenanceTask):
    """정책 평가 작업"""

    def __init__(self, batch_manager: BatchLifecycleManager):
        self.batch_manager = batch_manager

    @property
    def name(self) -> str:
        return "policy_evaluation"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """모든 문서에 대해 정책 평가 실행"""
        result = self.batch_manager.evaluate_all()
        logger.info(
            f"Policy evaluation completed: "
            f"{result['evaluated']} evaluated, "
            f"{result['state_changes']} state changes"
        )
        return result


class CleanupTask(MaintenanceTask):
    """정리 작업"""

    def __init__(self, batch_manager: BatchLifecycleManager):
        self.batch_manager = batch_manager

    @property
    def name(self) -> str:
        return "cleanup"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """삭제된 문서 정리"""
        deleted_ids = self.batch_manager.cleanup_deleted()
        logger.info(f"Cleanup completed: {len(deleted_ids)} documents removed")
        return {"deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}


class TierMigrationTask(MaintenanceTask):
    """계층 마이그레이션 작업"""

    def __init__(
        self,
        batch_manager: BatchLifecycleManager,
        from_tier: StorageTier,
        to_tier: StorageTier,
        limit: int = 100
    ):
        self.batch_manager = batch_manager
        self.from_tier = from_tier
        self.to_tier = to_tier
        self.limit = limit

    @property
    def name(self) -> str:
        return f"tier_migration_{self.from_tier.value}_to_{self.to_tier.value}"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """저장소 계층 마이그레이션"""
        migrated = self.batch_manager.migrate_tier(
            self.from_tier,
            self.to_tier,
            self.limit
        )
        logger.info(
            f"Tier migration completed: {len(migrated)} documents migrated"
        )
        return {"migrated_count": len(migrated), "migrated_ids": migrated}


class StatisticsTask(MaintenanceTask):
    """통계 수집 작업"""

    def __init__(self, manager: LifecycleManager):
        self.manager = manager

    @property
    def name(self) -> str:
        return "statistics_collection"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """통계 수집"""
        stats = self.manager.get_statistics()
        logger.info(f"Statistics collected: {stats['total_documents']} total")
        return stats


@dataclass
class SchedulerConfig:
    """스케줄러 설정"""
    check_interval_seconds: int = 60
    max_concurrent_tasks: int = 5
    default_timeout_seconds: int = 300
    enable_auto_start: bool = False


class LifecycleScheduler:
    """생명주기 스케줄러"""

    def __init__(
        self,
        lifecycle_manager: LifecycleManager,
        config: Optional[SchedulerConfig] = None
    ):
        self.lifecycle_manager = lifecycle_manager
        self.batch_manager = BatchLifecycleManager(lifecycle_manager)
        self.config = config or SchedulerConfig()

        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_handlers: Dict[str, MaintenanceTask] = {}
        self._results: List[TaskResult] = []
        self._running = False
        self._task_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_tasks
        )

        # 기본 작업 등록
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        """기본 유지보수 작업 등록"""
        # 정책 평가 (매일)
        self.register_task(
            ScheduledTask(
                task_id="daily_policy_evaluation",
                name="Daily Policy Evaluation",
                schedule_type=ScheduleType.DAILY,
                daily_time="02:00",
                priority=TaskPriority.NORMAL,
            ),
            PolicyEvaluationTask(self.batch_manager)
        )

        # 정리 작업 (매주)
        self.register_task(
            ScheduledTask(
                task_id="weekly_cleanup",
                name="Weekly Cleanup",
                schedule_type=ScheduleType.WEEKLY,
                weekly_day=0,  # 월요일
                daily_time="03:00",
                priority=TaskPriority.LOW,
            ),
            CleanupTask(self.batch_manager)
        )

        # 통계 수집 (매시간)
        self.register_task(
            ScheduledTask(
                task_id="hourly_statistics",
                name="Hourly Statistics",
                schedule_type=ScheduleType.INTERVAL,
                interval_seconds=3600,
                priority=TaskPriority.LOW,
            ),
            StatisticsTask(self.lifecycle_manager)
        )

    def register_task(
        self,
        scheduled_task: ScheduledTask,
        handler: MaintenanceTask
    ) -> None:
        """작업 등록"""
        scheduled_task.next_run = scheduled_task.calculate_next_run()
        self._tasks[scheduled_task.task_id] = scheduled_task
        self._task_handlers[scheduled_task.task_id] = handler
        logger.info(f"Task registered: {scheduled_task.task_id}")

    def unregister_task(self, task_id: str) -> bool:
        """작업 등록 해제"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            del self._task_handlers[task_id]
            logger.info(f"Task unregistered: {task_id}")
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """작업 활성화"""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """작업 비활성화"""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False

    async def run_task(
        self,
        task_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """작업 즉시 실행"""
        if task_id not in self._tasks:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                started_at=datetime.now(),
                error=f"Task not found: {task_id}"
            )

        scheduled_task = self._tasks[task_id]
        handler = self._task_handlers[task_id]
        context = context or {}

        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )

        async with self._task_semaphore:
            try:
                # 타임아웃 적용
                task_result = await asyncio.wait_for(
                    handler.execute(context),
                    timeout=scheduled_task.timeout_seconds
                )

                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.completed_at = datetime.now()
                result.duration_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

                # 성공 처리
                scheduled_task.run_count += 1
                scheduled_task.consecutive_failures = 0
                scheduled_task.last_run = datetime.now()
                scheduled_task.next_run = scheduled_task.calculate_next_run()

                handler.on_success(task_result)

            except asyncio.TimeoutError:
                result.status = TaskStatus.FAILED
                result.error = f"Task timed out after {scheduled_task.timeout_seconds}s"
                result.completed_at = datetime.now()
                scheduled_task.failure_count += 1
                scheduled_task.consecutive_failures += 1

            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = str(e)
                result.completed_at = datetime.now()
                scheduled_task.failure_count += 1
                scheduled_task.consecutive_failures += 1
                handler.on_failure(e)
                logger.error(f"Task failed: {task_id} - {e}")

        self._results.append(result)
        return result

    async def start(self) -> None:
        """스케줄러 시작"""
        if self._running:
            return

        self._running = True
        logger.info("Lifecycle scheduler started")

        while self._running:
            await self._check_and_run_tasks()
            await asyncio.sleep(self.config.check_interval_seconds)

    async def stop(self) -> None:
        """스케줄러 중지"""
        self._running = False
        logger.info("Lifecycle scheduler stopped")

    async def _check_and_run_tasks(self) -> None:
        """실행할 작업 확인 및 실행"""
        tasks_to_run = []

        for task_id, scheduled_task in self._tasks.items():
            if scheduled_task.should_run():
                tasks_to_run.append((task_id, scheduled_task.priority))

        # 우선순위순 정렬
        tasks_to_run.sort(key=lambda x: x[1].value, reverse=True)

        # 병렬 실행
        for task_id, _ in tasks_to_run:
            asyncio.create_task(self.run_task(task_id))

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        if task_id not in self._tasks:
            return None

        task = self._tasks[task_id]
        return {
            "task_id": task.task_id,
            "name": task.name,
            "enabled": task.enabled,
            "schedule_type": task.schedule_type.value,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "failure_count": task.failure_count,
            "consecutive_failures": task.consecutive_failures,
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """모든 작업 조회"""
        return [
            self.get_task_status(task_id)
            for task_id in self._tasks.keys()
        ]

    def get_recent_results(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """최근 실행 결과 조회"""
        results = self._results[-limit:]
        return [
            {
                "task_id": r.task_id,
                "status": r.status.value,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in reversed(results)
        ]


class QuickScheduler:
    """간단한 스케줄러 (단일 작업용)"""

    def __init__(self):
        self._callbacks: Dict[str, Callable] = {}
        self._intervals: Dict[str, int] = {}
        self._running = False

    def every(
        self,
        seconds: int,
        name: str,
        callback: Callable[[], Awaitable[None]]
    ) -> None:
        """주기적 작업 등록"""
        self._callbacks[name] = callback
        self._intervals[name] = seconds

    async def start(self) -> None:
        """스케줄러 시작"""
        self._running = True

        async def run_periodic(name: str, interval: int):
            while self._running:
                await asyncio.sleep(interval)
                try:
                    await self._callbacks[name]()
                except Exception as e:
                    logger.error(f"Periodic task {name} failed: {e}")

        tasks = [
            run_periodic(name, interval)
            for name, interval in self._intervals.items()
        ]

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """스케줄러 중지"""
        self._running = False
