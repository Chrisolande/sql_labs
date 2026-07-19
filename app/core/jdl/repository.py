import uuid
from collections import defaultdict
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.db.models.job import Job, JobDescription, JobSource


def _skills_text_for(title: str | None, skills: list[str] | None) -> str:
    parts = [title or ""]
    parts.extend(s for s in (skills or []) if s)
    return " ".join(parts).strip()


def _search_text_for(
    title: str | None,
    role: str | None,
    job_function: str | None,
    company_name: str | None,
    skills_text: str | None,
) -> str:
    parts = [
        title or "",
        role or "",
        job_function or "",
        company_name or "",
        skills_text or "",
    ]
    return " ".join(p for p in parts if p).strip()


async def upsert_job(
    db: AsyncSession, normalized, now_utc: datetime
) -> tuple[Job, bool]:
    values = normalized.model_dump(exclude={"source"}) | {"last_seen_at": now_utc}
    values["skills_text"] = _skills_text_for(
        values.get("title"), values.get("required_skills")
    )
    values["search_text"] = _search_text_for(
        values.get("title"),
        values.get("role"),
        values.get("job_function"),
        values.get("company_name"),
        values["skills_text"],
    )

    stmt = (
        pg_insert(Job)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["dedup_hash"],
            set_={
                "last_seen_at": now_utc,
                "status": "active",
                "skills_text": values["skills_text"],
                "search_text": values["search_text"],
                "required_skills": values.get("required_skills") or [],
                "title": values.get("title"),
            },
        )
        .returning(Job, text("xmax = 0 AS is_inserted"))
    )

    res = await db.execute(stmt)
    row = res.one()
    job, is_inserted = row[0], row[1]

    return job, is_inserted


async def upsert_job_source(
    db: AsyncSession, job: Job, normalized, now_utc: datetime
) -> None:
    stmt = select(JobSource).where(
        JobSource.source_name == normalized.source.source_name,
        JobSource.source_job_id == normalized.source.source_job_id,
    )
    source = (await db.execute(stmt)).scalar_one_or_none()

    if source is None:
        source = JobSource(
            id=str(uuid.uuid4()),
            job_id=job.id,
            source_name=normalized.source.source_name,
            source_job_id=normalized.source.source_job_id,
            source_url=normalized.source.source_url,
            last_checked_at=now_utc,
            unconfirmed_count=0,
        )
        db.add(source)
    else:
        source.last_checked_at = now_utc
        source.unconfirmed_count = 0
        if normalized.source.source_url:
            source.source_url = normalized.source.source_url

    await db.flush()


async def get_job_source(db: AsyncSession, job_id: str) -> JobSource | None:
    return (
        await db.execute(
            select(JobSource)
            .where(JobSource.job_id == job_id)
            .order_by(JobSource.last_checked_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def upsert_job_description(
    db: AsyncSession, job_id: str, cleaned_text: str
) -> JobDescription:
    stmt = (
        pg_insert(JobDescription)
        .values(
            job_id=job_id,
            cleaned_text=cleaned_text,
            fetched_at=datetime.now(UTC),
        )
        .on_conflict_do_nothing(index_elements=["job_id"])
    )
    await db.execute(stmt)
    await db.flush()
    return await db.get(JobDescription, job_id)


async def close_stale_jobs(
    db: AsyncSession,
    source_name: str,
    seen_source_job_ids: set[str],
    unconfirmed_limit: int,
) -> int:
    stmt_unseen = select(JobSource).where(JobSource.source_name == source_name)
    if seen_source_job_ids:
        stmt_unseen = stmt_unseen.where(
            JobSource.source_job_id.not_in(seen_source_job_ids)
        )

    unseen_sources = (await db.execute(stmt_unseen)).scalars().all()
    if not unseen_sources:
        return 0

    for src in unseen_sources:
        src.unconfirmed_count += 1

    jobs_to_check = {src.job_id for src in unseen_sources}

    stmt_all_srcs = select(JobSource).where(JobSource.job_id.in_(jobs_to_check))
    all_srcs = (await db.execute(stmt_all_srcs)).scalars().all()

    srcs_by_job: dict[str, list[JobSource]] = defaultdict(list)
    for s in all_srcs:
        srcs_by_job[s.job_id].append(s)

    jobs_to_close = [
        job_id
        for job_id, srcs in srcs_by_job.items()
        if all(s.unconfirmed_count >= unconfirmed_limit for s in srcs)
    ]

    if not jobs_to_close:
        return 0

    stmt_jobs = select(Job).where(Job.id.in_(jobs_to_close), Job.status != "closed")
    jobs = (await db.execute(stmt_jobs)).scalars().all()
    jobs_closed = 0
    for job in jobs:
        job.status = "closed"
        jobs_closed += 1

    return jobs_closed


async def list_jobs_repo(
    db: AsyncSession, keyword: str | None = None, page: int = 1, page_size: int = 20
) -> tuple[int, list[Job]]:
    from sqlalchemy import func

    offset = (page - 1) * page_size
    base = select(Job).where(Job.status == "active")
    count_stmt = select(func.count()).select_from(Job).where(Job.status == "active")

    if keyword:
        pattern = f"%{keyword}%"
        base = base.where(Job.title.ilike(pattern) | Job.company_name.ilike(pattern))
        count_stmt = count_stmt.where(
            Job.title.ilike(pattern) | Job.company_name.ilike(pattern)
        )

    total = (await db.execute(count_stmt)).scalar_one() or 0
    rows = (
        (
            await db.execute(
                base.order_by(Job.created_at.desc()).offset(offset).limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return total, list(rows)


async def get_job_repo(db: AsyncSession, job_id: str) -> Job | None:
    from sqlalchemy.orm import selectinload

    stmt = (
        select(Job)
        .options(selectinload(Job.description), selectinload(Job.sources))
        .where(Job.id == job_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()
