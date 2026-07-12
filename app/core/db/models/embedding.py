import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Index, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base


class EntityType(enum.StrEnum):
    job = "job"
    career_memory = "career_memory"


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entitytype"), nullable=False
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    embedding_input_hash: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )
    vector: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    valid_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW() + INTERVAL '7 days'"),
        nullable=False,
    )


# HNSW index for cosine distance vector search
Index(
    "ix_embeddings_vector_hnsw",
    Embedding.vector,
    postgresql_using="hnsw",
    postgresql_ops={"vector": "vector_cosine_ops"},
)
