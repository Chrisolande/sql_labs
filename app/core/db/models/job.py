import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dedup_hash: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    domain_name: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    job_function: Mapped[str | None] = mapped_column(String, nullable=True)
    seniority: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    employment_type: Mapped[str | None] = mapped_column(String, nullable=True)
    remote_type: Mapped[str | None] = mapped_column(String, nullable=True)
    locations: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    countries: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    required_skills: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )

    skills_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    employee_count: Mapped[str | None] = mapped_column(String, nullable=True)
    funding: Mapped[str | None] = mapped_column(String, nullable=True)
    company_summary: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, server_default="active", nullable=False)
    posted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    fallback_source: Mapped[str | None] = mapped_column(String, nullable=True)

    sources = relationship("JobSource", back_populates="job")
    description = relationship("JobDescription", back_populates="job", uselist=False)

    __table_args__ = (
        Index(
            "idx_jobs_required_skills_gin", "required_skills", postgresql_using="gin"
        ),
        Index("idx_jobs_locations_gin", "locations", postgresql_using="gin"),
        Index("idx_jobs_countries_gin", "countries", postgresql_using="gin"),
        Index("idx_jobs_seniority_gin", "seniority", postgresql_using="gin"),
    )


class JobSource(Base):
    __tablename__ = "job_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_job_id: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[str] = mapped_column(String, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    unconfirmed_count: Mapped[int] = mapped_column(
        Integer, server_default=text("0"), nullable=False
    )

    job = relationship("Job", back_populates="sources")

    __table_args__ = (
        UniqueConstraint(
            "source_name", "source_job_id", name="uq_job_sources_name_job_id"
        ),
    )


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )

    job = relationship("Job", back_populates="description")
