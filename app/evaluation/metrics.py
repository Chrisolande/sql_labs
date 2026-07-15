from collections import defaultdict
from functools import cache, partial
from typing import Any

import ir_measures
from ir_measures import RR, R, nDCG
from langsmith.schemas import Example, Run
from loguru import logger

from app.evaluation.constants import NO_MATCH_STYLE

NDCG_REGRESSION_THRESHOLD: float = 0.05
MRR_REGRESSION_THRESHOLD: float = 0.05


MEASURES = [nDCG @ 10, nDCG @ 20, RR, R @ 10, R @ 20]

MEASURE_KEYS: dict = {
    nDCG @ 10: "nDCG@10",
    nDCG @ 20: "nDCG@20",
    RR: "RR",
    R @ 10: "Recall@10",
    R @ 20: "Recall@20",
}

METRIC_ORDER: tuple[str, ...] = ("nDCG@10", "nDCG@20", "RR", "Recall@10", "Recall@20")


def query_style_of(example: Any) -> str:
    meta = getattr(example, "metadata", None) or {}
    if isinstance(meta, dict) and meta.get("query_style"):
        return str(meta["query_style"])
    inputs = getattr(example, "inputs", None) or {}
    if isinstance(inputs, dict) and inputs.get("query_style"):
        return str(inputs["query_style"])
    return "unknown"


def expected_job_id_of(example: Any) -> str | None:
    outputs = getattr(example, "outputs", None) or {}
    raw = outputs.get("expected_job_id")
    if raw is None or raw == "" or str(raw).lower() == "none":
        return None
    return str(raw)


def is_matching_example(example: Any) -> bool:
    if query_style_of(example) == NO_MATCH_STYLE:
        return False
    return expected_job_id_of(example) is not None


def ranked_ids_to_run(
    qid: str, ranked_job_ids: list[str]
) -> dict[str, dict[str, float]]:
    return {
        qid: {
            job_id: float(len(ranked_job_ids) - rank)
            for rank, job_id in enumerate(ranked_job_ids)
        }
    }


def scores_to_run(
    qid: str, id_to_score: dict[str, float]
) -> dict[str, dict[str, float]]:
    return {qid: {str(k): float(v) for k, v in id_to_score.items()}}


def build_qrels_and_run_from_rankings(examples, rankings):
    qrels = {}
    run = {}
    skipped: list[str] = []

    for example in examples:
        qid = str(example.id)
        if not is_matching_example(example):
            logger.debug("Skipping qid={} style={}", qid[:8], query_style_of(example))
            skipped.append(qid)
            continue

        expected = expected_job_id_of(example)
        assert expected is not None
        qrels[qid] = {expected: 1}

        ranking = rankings.get(qid, {})
        if isinstance(ranking, dict):
            run[qid] = {str(k): float(v) for k, v in ranking.items()}
        else:
            run[qid] = ranked_ids_to_run(qid, ranking)[qid]

        logger.debug(
            "qid={} expected={!r} expected_in_run={} run_size={}",
            qid[:8],
            expected,
            expected in run[qid],
            len(run[qid]),
        )

    return qrels, run, skipped


def to_friendly(raw: dict) -> dict[str, float]:
    return {MEASURE_KEYS.get(m, str(m)): float(v or 0.0) for m, v in raw.items()}


def compute_aggregate_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    measures: list | None = None,
) -> dict[str, float]:
    if not qrels:
        return dict.fromkeys(MEASURE_KEYS.values(), 0.0)
    measures = measures or MEASURES
    logger.debug("compute_aggregate_metrics: n_qrels={} n_run={}", len(qrels), len(run))
    raw = ir_measures.calc_aggregate(measures, qrels, run)
    return to_friendly(raw)


def compute_metrics_by_style(
    examples: list[Any],
    rankings: dict[str, list[str] | dict[str, float]],
) -> dict[str, Any]:
    by_style: dict[str, list[Any]] = defaultdict(list)
    for ex in examples:
        by_style[query_style_of(ex)].append(ex)

    overall_qrels, overall_run, skipped = build_qrels_and_run_from_rankings(
        examples, rankings
    )
    result: dict[str, Any] = {
        "overall": compute_aggregate_metrics(overall_qrels, overall_run),
        "n_matching": len(overall_qrels),
        "n_skipped_no_match": len(skipped),
        "by_style": {},
        "no_match": {},
    }

    for style, style_examples in sorted(by_style.items()):
        if style == NO_MATCH_STYLE:
            continue
        qrels, run, _ = build_qrels_and_run_from_rankings(style_examples, rankings)
        if not qrels:
            continue
        result["by_style"][style] = {
            "n": len(qrels),
            **compute_aggregate_metrics(qrels, run),
        }

    no_match_examples = by_style.get(NO_MATCH_STYLE, [])
    if no_match_examples:
        hit_flags = [bool(rankings.get(str(ex.id)) or {}) for ex in no_match_examples]
        n = len(hit_flags)
        any_hit_rate = sum(hit_flags) / n if n else 0.0
        result["no_match"] = {
            "n": n,
            "empty_result_rate": 1 - any_hit_rate if n else 0.0,
            "any_hit_rate": any_hit_rate,
            "note": (
                "no_match queries have no relevant golden job; they are excluded "
                "from nDCG/RR/Recall aggregates. any_hit_rate is informational only "
                "when the index contains more than the golden set."
            ),
        }

    return result


def per_query_metrics(
    qid: str,
    expected_job_id: str | None,
    ranked_job_ids: list[str],
    measures: list | None = None,
) -> dict[str, float] | None:
    if expected_job_id is None:
        return None
    qrels = {qid: {expected_job_id: 1}}
    run = ranked_ids_to_run(qid, ranked_job_ids)
    raw = ir_measures.calc_aggregate(measures or MEASURES, qrels, run)
    return to_friendly(raw)


def fmt_scores(scores: dict, pad: str) -> list[str]:
    return [f"{pad}{key}: {scores[key]:.4f}" for key in METRIC_ORDER if key in scores]


def format_metrics(report: dict[str, Any], indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    overall = report.get("overall", {})
    lines.append(f"{pad}overall (n_matching={report.get('n_matching', 0)}):")
    lines.extend(fmt_scores(overall, pad + "  "))

    by_style = report.get("by_style") or {}
    if by_style:
        lines.append(f"{pad}by_style:")
        for style, scores in by_style.items():
            n = scores.get("n", "?")
            lines.append(f"{pad}  [{style}] n={n}")
            lines.extend(fmt_scores(scores, pad + "    "))

    no_match = report.get("no_match") or {}
    if no_match:
        lines.append(
            f"{pad}no_match (n={no_match.get('n', 0)}): "
            f"empty_result_rate={no_match.get('empty_result_rate', 0):.4f}, "
            f"any_hit_rate={no_match.get('any_hit_rate', 0):.4f}"
        )
    return "\n".join(lines)


def extract_ranked_job_ids(run: Run) -> list[str]:
    outputs = run.outputs or {}
    ranked_jobs = outputs.get("ranked_jobs") or []
    if ranked_jobs and isinstance(ranked_jobs[0], dict):
        return [str(job["id"]) for job in ranked_jobs]
    return [str(job_id) for job_id in outputs.get("ranked_job_ids", [])]


@cache
def metrics_for_ranked(
    qid: str, expected: str, ranked: tuple[str, ...]
) -> dict[str, float] | None:
    return per_query_metrics(qid, expected, list(ranked))


def metrics_for(run: Run, example: Example) -> dict[str, float] | None:
    expected = expected_job_id_of(example)
    if expected is None:
        return None
    ranked = tuple(extract_ranked_job_ids(run))
    return metrics_for_ranked(str(example.id), expected, ranked)


def score(run: Run, example: Example, metric_key: str, feedback_key: str) -> dict:
    try:
        outputs = run.outputs or {}
        if not outputs.get("ranked_jobs") and not outputs.get("ranked_job_ids"):
            return {"key": feedback_key, "score": 0.0}

        m = metrics_for(run, example)
        if m is None:
            return {
                "key": feedback_key,
                "score": None,
                "comment": f"skipped ({query_style_of(example)} / no expected_job_id)",
            }

        return {"key": feedback_key, "score": float(m.get(metric_key, 0.0))}
    except Exception as e:
        logger.error(f"Error computing {feedback_key}: {e}")
        return {"key": feedback_key, "score": 0.0, "comment": str(e)}


ndcg_at_10 = partial(score, metric_key="nDCG@10", feedback_key="ndcg_at_10")
ndcg_at_20 = partial(score, metric_key="nDCG@20", feedback_key="ndcg_at_20")
mrr = partial(score, metric_key="RR", feedback_key="mrr")
recall_at_10 = partial(score, metric_key="Recall@10", feedback_key="recall_at_10")
recall_at_20 = partial(score, metric_key="Recall@20", feedback_key="recall_at_20")

RETRIEVAL_EVALUATORS = [ndcg_at_10, ndcg_at_20, mrr, recall_at_10, recall_at_20]
