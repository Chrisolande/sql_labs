import uuid

import httpx
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.batch import process_in_batches
from app.core.db.models.job import Job, JobDescription, JobSource
from app.core.retry import default_retry

JINA_PREFIX = "https://r.jina.ai/"

MAX_CONCURRENT_REQUESTS = 3
REQUESTS_PER_SECOND = 2


@default_retry(initial=2, max_wait=60)
async def fetch_jina_content(
    client: httpx.AsyncClient,
    source_url: str,
) -> str:
    jina_url = f"{JINA_PREFIX}{source_url}"

    response = await client.get(jina_url)

    if response.status_code == 429:
        logger.warning("Rate limited by Jina: %s", source_url)
        response.raise_for_status()

    response.raise_for_status()

    return response.text


async def populate_job_descriptions(
    db: AsyncSession,
    batch_size: int = 10,
    job_ids: list[str] | None = None,
) -> None:
    stmt = (
        select(
            Job.id,
            JobSource.source_url,
        )
        .join(JobSource, JobSource.job_id == Job.id)
        .outerjoin(
            JobDescription,
            JobDescription.job_id == Job.id,
        )
        .where(JobDescription.job_id.is_(None))
        .distinct(Job.id)
    )

    if job_ids:
        uuids = [uuid.UUID(jid) for jid in job_ids]
        stmt = stmt.where(Job.id.in_(uuids))

    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        logger.info("No jobs require description fetching")
        return

    logger.info("Found %d jobs requiring descriptions", len(rows))

    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=True,
    ) as client:
        responses = await process_in_batches(
            items=list(rows),
            processor=lambda row: fetch_jina_content(client, row.source_url),
            batch_size=batch_size,
            max_concurrency=MAX_CONCURRENT_REQUESTS,
        )

        descriptions: list[JobDescription] = []

        for (job_id, source_url), response in zip(rows, responses, strict=False):
            if isinstance(response, Exception):
                logger.warning(
                    "Failed fetching job %s (%s): %s",
                    job_id,
                    source_url,
                    response,
                )
                continue

            descriptions.append(
                JobDescription(
                    job_id=job_id,
                    cleaned_text=response,
                )
            )

        if descriptions:
            db.add_all(descriptions)
            await db.commit()

            logger.info(
                "Inserted %d descriptions",
                len(descriptions),
            )

    logger.info("Finished populating job descriptions")
