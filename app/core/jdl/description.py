import uuid
from itertools import batched

import httpx
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.batch import process_in_batches
from app.core.config.settings import settings
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
        logger.warning("Rate limited by Jina: {}", source_url)
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

    logger.info("Found {} jobs requiring descriptions", len(rows))

    headers = {}
    if settings.jina_api_key:
        headers["Authorization"] = f"Bearer {settings.jina_api_key}"

    async with httpx.AsyncClient(
        headers=headers,
        timeout=60,
        follow_redirects=True,
    ) as client:
        for batch in batched(rows, batch_size, strict=False):
            logger.info("Processing batch of {} jobs", len(batch))
            responses = await process_in_batches(
                items=list(batch),
                processor=lambda row: fetch_jina_content(client, row.source_url),
                batch_size=len(batch),
                max_concurrency=MAX_CONCURRENT_REQUESTS,
                rate_per_second=REQUESTS_PER_SECOND,
            )

            descriptions: list[JobDescription] = []

            for (job_id, source_url), response in zip(batch, responses, strict=False):
                if isinstance(response, Exception):
                    logger.warning(
                        "Failed fetching job {} ({}): {}",
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
                    "Inserted {} descriptions for this batch",
                    len(descriptions),
                )

    logger.info("Finished populating job descriptions")
