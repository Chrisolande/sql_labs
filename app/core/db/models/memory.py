import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base


class MemoryEntityType(enum.StrEnum):
    project = "project"
    skill = "skill"
    achievement = "achievement"
    education = "education"
    employment_history = "employment_history"


class CareerMemory(Base):
    __tablename__ = "career_memory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    entity_type: Mapped[MemoryEntityType] = mapped_column(
        Enum(MemoryEntityType, name="memoryentitytype"), nullable=False, index=True
    )
    fact_key: Mapped[str] = mapped_column(nullable=False, default="default")
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
