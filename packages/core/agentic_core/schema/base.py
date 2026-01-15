"""
Base Schema Classes

공통 베이스 클래스 및 믹스인 정의
"""

from datetime import datetime
from typing import Optional, Dict, Any, TypeVar, Type
from pydantic import BaseModel, Field
import uuid


T = TypeVar("T", bound="SchemaBase")


class SchemaBase(BaseModel):
    """모든 스키마의 베이스 클래스"""

    class Config:
        # JSON 직렬화 시 enum을 값으로
        use_enum_values = True
        # 추가 필드 허용 안함
        extra = "forbid"
        # datetime을 ISO 포맷으로
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return self.model_dump()

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """딕셔너리에서 생성"""
        return cls.model_validate(data)


class Identifiable(SchemaBase):
    """고유 ID를 가진 엔티티"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identifiable):
            return False
        return self.id == other.id


class Timestamped(SchemaBase):
    """생성/수정 시간을 추적하는 엔티티"""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def touch(self) -> None:
        """수정 시간 갱신"""
        object.__setattr__(self, "updated_at", datetime.now())


class IdentifiableTimestamped(Identifiable, Timestamped):
    """ID + 시간 추적을 모두 가진 엔티티"""
    pass


class Scorable(SchemaBase):
    """점수/신뢰도를 가진 엔티티"""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Provenanced(SchemaBase):
    """출처 추적이 가능한 엔티티"""

    source_ids: list[str] = Field(default_factory=list)
    extraction_method: Optional[str] = None

    def add_source(self, source_id: str) -> None:
        """출처 추가"""
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)
