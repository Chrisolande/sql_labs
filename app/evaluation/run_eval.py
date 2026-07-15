import asyncio
import sys
from statistics import mean

from langsmith import Client, aevaluate
from loguru import logger

from app.core.config.settings import settings
from app.evaluation.metrics import (
    MRR_REGRESSION_THRESHOLD,
    NDCG_REGRESSION_THRESHOLD,
    RETRIEVAL_EVALUATORS,
)
from app.evaluation.search import (
    DEFAULT_SEARCH_PARAMS,
    RETRIEVAL_DATASET,
    retrieval_target,
)

# Metric key: acceptable relative-drop threshold
REGRESSION_THRESHOLDS: dict[str, float] = {
    "ndcg_at_10": NDCG_REGRESSION_THRESHOLD,
    "mrr": MRR_REGRESSION_THRESHOLD,
}


def relative_drop(prior: float, current: float) -> float | None:
    if prior <= 0:
        return None
    return (prior - current) / prior


def _feedback_avg(client: Client, project_name: str, metric_name: str) -> float:
    try:
        run_ids = [r.id for r in client.list_runs(project_name=project_name)]
        if not run_ids:
            return 0.0
        scores = [
            float(fb.score)
            for fb in client.list_feedback(run_ids=run_ids)
            if fb.key == metric_name and fb.score is not None
        ]
        return mean(scores) if scores else 0.0
    except Exception as exc:
        logger.warning("Error fetching feedback for {}: {}", project_name, exc)
        return 0.0


def check_retrieval_regression(client: Client, current_project_name: str) -> bool:
    """Return True if any metric regresses beyond its configured threshold."""
    dataset = next(client.list_datasets(dataset_name=RETRIEVAL_DATASET), None)
    if dataset is None:
        logger.info("No dataset found — skipping regression check.")
        return False

    # Most-recent first, current project is at index 0 after sorting.
    projects = sorted(
        client.list_projects(reference_dataset_id=dataset.id),
        key=lambda p: p.start_time or p.created_at,
        reverse=True,
    )
    names = {p.name: p for p in projects}

    if current_project_name not in names:
        logger.warning(
            "Current project {!r} not found among experiments — skipping.",
            current_project_name,
        )
        return False

    prior = next((p for p in projects if p.name != current_project_name), None)
    if prior is None:
        logger.info("No prior experiment to compare against.")
        return False

    has_regression = False
    for metric_name, threshold in REGRESSION_THRESHOLDS.items():
        current_score = _feedback_avg(client, current_project_name, metric_name)
        prior_score = _feedback_avg(client, prior.name, metric_name)
        logger.info(
            "Current {}: {:.4f}  Prior {}: {:.4f}",
            metric_name,
            current_score,
            metric_name,
            prior_score,
        )
        drop = relative_drop(prior_score, current_score)
        if drop is not None and drop > threshold:
            logger.error(
                "Regression in {}! Drop {:.1f}% > threshold {:.1f}%",
                metric_name,
                drop * 100,
                threshold * 100,
            )
            has_regression = True

    return has_regression


async def run_evaluations() -> None:
    if not settings.langsmith_api_key:
        logger.error("LANGSMITH_API_KEY not set. Cannot run evaluations.")
        sys.exit(1)

    client = Client()
    params = DEFAULT_SEARCH_PARAMS
    logger.info(
        "Running retrieval evaluation on dataset={} with params={}",
        RETRIEVAL_DATASET,
        params.as_dict(),
    )

    async def target(inputs: dict) -> dict:
        return await retrieval_target(inputs, params=params)

    results = await aevaluate(
        target,
        data=RETRIEVAL_DATASET,
        evaluators=RETRIEVAL_EVALUATORS,
        experiment_prefix="hybrid-retrieval",
        metadata={"search_params": params.as_dict(), "dataset": RETRIEVAL_DATASET},
    )

    has_regression = False
    try:
        has_regression = check_retrieval_regression(client, results.experiment_name)
    except Exception as exc:
        logger.warning("Error comparing retrieval metrics: {}", exc)

    logger.info("Evaluations completed.")
    if has_regression:
        sys.exit(1)


def main() -> None:
    asyncio.run(run_evaluations())


if __name__ == "__main__":
    main()
