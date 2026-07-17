import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from itertools import batched

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config.settings import settings
from app.core.db.models.embedding import Embedding, EntityType
from app.core.db.models.job import Job
from app.core.jdl.schemas import JobEmbeddingDocument
from app.core.retry import default_retry

EMBEDDING_TEXT_VERSION = 1
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_MAX_CONCURRENCY = 3


def get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=1024,
        api_key=settings.google_api_key,
    )


def build_job_embedding_text(job: Job) -> str:
    doc = JobEmbeddingDocument(
        title=job.title,
        company=job.company_name,
        role=job.role,
        function=job.job_function,
        seniority=job.seniority,
        employment=job.employment_type,
        remote=job.remote_type,
        locations=job.locations,
        skills=job.required_skills,
        description=getattr(job.description, "cleaned_text", None),
        company_summary=job.company_summary,
    )
    return "\n".join(f"{k}:{v}" for k, v in doc.model_dump(exclude_none=True).items())


def compute_job_embedding_hash(job: Job) -> str:
    text_content = build_job_embedding_text(job)
    hash_input = f"v{EMBEDDING_TEXT_VERSION}:{text_content}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


async def fetch_active_jobs(
    session: AsyncSession, job_ids: list[int] | None = None
) -> list[Job]:
    stmt = (
        select(Job).options(selectinload(Job.description)).where(Job.status == "active")
    )
    if job_ids is not None:
        stmt = stmt.where(Job.id.in_(job_ids))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def fetch_existing_embeddings(
    session: AsyncSession, job_ids: list[int]
) -> dict[int, Embedding]:
    stmt = select(Embedding).where(
        Embedding.entity_type == EntityType.job, Embedding.entity_id.in_(job_ids)
    )
    result = await session.execute(stmt)
    return {emb.entity_id: emb for emb in result.scalars().all()}


def needs_embedding_update(
    emb: Embedding | None, fresh_hash: str, now: datetime
) -> bool:
    if not emb:
        return True
    return emb.valid_until < now or emb.embedding_input_hash != fresh_hash


def collect_stale_jobs(jobs: list[Job], emb_map: dict[int, Embedding], now: datetime):
    jobs_to_update, texts_to_embed = [], []
    for job in jobs:
        fresh_hash = compute_job_embedding_hash(job)
        fresh_text = build_job_embedding_text(job)
        emb = emb_map.get(job.id)
        if needs_embedding_update(emb, fresh_hash, now):
            jobs_to_update.append((job, emb, fresh_hash, fresh_text))
            texts_to_embed.append(fresh_text)
    return jobs_to_update, texts_to_embed


async def embed_texts_in_batches(
    client: GoogleGenerativeAIEmbeddings,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_concurrency: int = EMBEDDING_MAX_CONCURRENCY,
) -> list[list[float]]:
    batches = list(batched(texts, batch_size, strict=False))
    semaphore = asyncio.Semaphore(max_concurrency)

    @default_retry(initial=1, max_wait=30)
    async def embed_one_batch(batch: tuple[str, ...]) -> list[list[float]]:
        async with semaphore:
            return await client.aembed_documents(list(batch))

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(embed_one_batch(b)) for b in batches]

    batch_results = [task.result() for task in tasks]

    vectors: list[list[float]] = []
    for batch_vectors in batch_results:
        vectors.extend(batch_vectors)
    return vectors


def apply_embedding_update(
    job: Job,
    emb: Embedding | None,
    fresh_hash: str,
    vector: list[float],
    now: datetime,
    session: AsyncSession,
):
    valid_until = now + timedelta(days=7)
    if not emb:
        emb = Embedding(
            entity_type=EntityType.job,
            entity_id=job.id,
            embedding_input_hash=fresh_hash,
            vector=vector,
            model_name="gemini-embedding-2",
            created_at=now,
            valid_until=valid_until,
        )
        session.add(emb)
    else:
        emb.embedding_input_hash = fresh_hash
        emb.vector = vector
        emb.model_name = "gemini-embedding-2"
        emb.created_at = now
        emb.valid_until = valid_until


async def refresh_stale_embeddings(session: AsyncSession) -> None:
    jobs = await fetch_active_jobs(session)
    if not jobs:
        return

    job_ids = [job.id for job in jobs]
    emb_map = await fetch_existing_embeddings(session, job_ids)
    now = datetime.now(UTC)

    jobs_to_update, texts_to_embed = collect_stale_jobs(jobs, emb_map, now)
    if not jobs_to_update:
        return

    client = get_embeddings_client()
    vectors = await embed_texts_in_batches(client, texts_to_embed)

    for (job, emb, fresh_hash, _), vector in zip(jobs_to_update, vectors, strict=False):
        apply_embedding_update(job, emb, fresh_hash, vector, now, session)
    await session.commit()
