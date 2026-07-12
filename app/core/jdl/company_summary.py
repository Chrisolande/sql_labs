from langchain_tavily import TavilySearch
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.batch import process_in_batches
from app.core.config.settings import settings
from app.core.db.base import async_session
from app.core.db.models.job import Job
from app.core.retry import default_retry


def get_tavily(include_answer: bool = True, k: int = 5) -> TavilySearch:
    return TavilySearch(
        include_answer=include_answer,
        k=k,
        tavily_api_key=settings.tavily_api_key,
    )


@default_retry(initial=2, max_wait=60)
async def fetch_company_summary(
    tavily: TavilySearch,
    company_name: str,
) -> str | None:
    response = await tavily.ainvoke(
        f"Provide a concise overview of the company {company_name}."
    )
    return response.get("answer")


async def _backfill_and_fetch(
    db: AsyncSession,
    company_filter: set[str] | None = None,
    batch_size: int = 10,
) -> None:
    source_job = aliased(Job)

    backfill_where = [Job.company_summary.is_(None)]
    if company_filter is not None:
        backfill_where.append(Job.company_name.in_(company_filter))

    backfill_stmt = (
        update(Job)
        .values(
            company_summary=(
                select(source_job.company_summary)
                .where(
                    source_job.company_name == Job.company_name,
                    source_job.company_summary.is_not(None),
                )
                .limit(1)
                .scalar_subquery()
            )
        )
        .where(*backfill_where)
    )

    backfilled = (await db.execute(backfill_stmt)).rowcount or 0

    if backfilled:
        logger.info(
            "Copied existing summaries onto %d job rows (no API cost)",
            backfilled,
        )

    select_where = [Job.company_summary.is_(None)]
    if company_filter is not None:
        select_where.append(Job.company_name.in_(company_filter))

    result = await db.execute(select(Job.company_name).where(*select_where).distinct())
    missing = result.scalars().all()

    if not missing:
        return

    tavily = get_tavily()

    responses = await process_in_batches(
        items=list(missing),
        processor=lambda name: fetch_company_summary(tavily, name),
        batch_size=batch_size,
        max_concurrency=batch_size,
    )

    for company_name, response in zip(missing, responses, strict=False):
        if isinstance(response, Exception):
            logger.warning(
                "Failed to summarize %s: %s",
                company_name,
                response,
            )
            continue

        if not response:
            continue

        await db.execute(
            update(Job)
            .where(Job.company_name == company_name)
            .values(company_summary=response)
        )

        logger.info("Summarized: %s", company_name)

    await db.commit()


async def populate_company_summaries(batch_size: int = 10) -> None:
    async with async_session() as db:
        await _backfill_and_fetch(db, company_filter=None, batch_size=batch_size)


async def populate_for_companies(
    db: AsyncSession,
    company_names: set[str],
    batch_size: int = 10,
) -> None:
    if not company_names:
        return
    await _backfill_and_fetch(db, company_filter=company_names, batch_size=batch_size)
