# scripts/jdl_daemon.py
import asyncio
from datetime import UTC, datetime

from loguru import logger

from app.core.config.settings import settings
from app.core.db.base import async_session
from app.core.db.models.ingestion_run import IngestionRun
from app.core.jdl.discovery import run_discovery
from app.core.jdl.repository import close_stale_jobs
from app.core.jdl.seed_criteria import DISCOVERY_SEED_CRITERIA

INTERVAL_SECONDS = 43200
SOURCE_NAME = "primary"


async def run_ingestion_cycle() -> None:
    all_seen_ids: set[str] = set()
    totals = {"created": 0, "updated": 0, "closed": 0}
    failed_seeds: list[dict] = []
    error: str | None = None

    async with async_session() as audit_db:
        run = IngestionRun(started_at=datetime.now(UTC), status="running")
        audit_db.add(run)
        await audit_db.commit()
        await audit_db.refresh(run)
    run_id = run.id

    try:
        for criteria in DISCOVERY_SEED_CRITERIA:
            seed_dump = criteria.model_dump(exclude_none=True)
            try:
                async with async_session() as db:
                    result = await run_discovery(
                        db=db,
                        criteria=criteria,
                        source_name=SOURCE_NAME,
                        seen_source_job_ids=all_seen_ids,
                        close_stale=False,
                    )
                totals["created"] += result.jobs_created
                totals["updated"] += result.jobs_updated
                logger.info(
                    "seed={} created={} updated={}",
                    seed_dump,
                    result.jobs_created,
                    result.jobs_updated,
                )
            except Exception as exc:
                logger.exception("seed failed: {}", seed_dump)
                failed_seeds.append({"seed": seed_dump, "error": str(exc)})
                continue

        async with async_session() as db:
            totals["closed"] = await close_stale_jobs(
                db,
                SOURCE_NAME,
                all_seen_ids,
                settings.discovery_unconfirmed_limit,
            )
            await db.commit()
        logger.info(
            "ingestion cycle closed={} seen_sources={}",
            totals["closed"],
            len(all_seen_ids),
        )
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if error:
            status = "failed"
        elif not failed_seeds:
            status = "success"
        elif len(failed_seeds) == len(DISCOVERY_SEED_CRITERIA):
            status = "failed"
        else:
            status = "partial"

        async with async_session() as audit_db:
            run = await audit_db.get(IngestionRun, run_id)
            run.finished_at = datetime.now(UTC)
            run.status = status
            run.created = totals["created"]
            run.updated = totals["updated"]
            run.closed = totals["closed"]
            run.failed_seeds = failed_seeds
            run.error = error
            await audit_db.commit()


async def daemon_loop() -> None:
    while True:
        try:
            await run_ingestion_cycle()
        except Exception:
            logger.exception("ingestion cycle failed entirely")
        await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(daemon_loop())
