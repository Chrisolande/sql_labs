import asyncio
import json
from typing import Annotated

from langsmith import Client
from loguru import logger
from typer import Option, Typer

from app.evaluation.metrics import (
    compute_metrics_by_style,
    expected_job_id_of,
    format_metrics,
    per_query_metrics,
    query_style_of,
)
from app.evaluation.search import (
    DEFAULT_SEARCH_PARAMS,
    RETRIEVAL_DATASET,
    SearchParams,
    precompute_query_embeddings,
    search_jobs_with_embedding,
)

app = Typer(
    name="run_local",
    help="Offline hybrid-search IR evaluation for CareerPilot.",
    no_args_is_help=False,
    add_completion=False,
)


async def _collect_rankings(
    examples: list,
    query_embeddings: dict[str, list[float]],
    params: SearchParams,
) -> dict[str, dict[str, float]]:
    from app.core.db.base import (
        async_session,
    )

    rankings: dict[str, dict[str, float]] = {}
    async with async_session() as db:
        for example in examples:
            qid = str(example.id)
            query_text = example.inputs["query"]
            embedding = query_embeddings[query_text]
            try:
                rows = await search_jobs_with_embedding(
                    db, query_text, embedding, params
                )
                rankings[qid] = {str(r["id"]): float(r["rrf_score"]) for r in rows}
            except Exception:
                logger.exception("search failed for query_id=%s", qid)
                rankings[qid] = {}
    return rankings


async def _evaluate_once(
    examples: list,
    query_embeddings: dict[str, list[float]],
    params: SearchParams,
) -> dict:
    rankings = await _collect_rankings(examples, query_embeddings, params)
    report = compute_metrics_by_style(examples, rankings)
    report["params"] = params.as_dict()
    return report


def _print_per_query(
    examples: list,
    rankings: dict[str, dict[str, float]],
) -> None:
    print("\n--- per-query ---")
    for example in examples:
        qid = str(example.id)
        style = query_style_of(example)
        expected = expected_job_id_of(example)
        ranking = rankings.get(qid, {})
        ordered = [
            doc_id
            for doc_id, _ in sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
        ]
        query = example.inputs.get("query", "")
        if expected is None:
            print(f"[{style}] no_match  hits={len(ordered)}  q={query!r}")
            continue
        m = per_query_metrics(qid, expected, ordered) or {}
        rank = ordered.index(expected) + 1 if expected in ordered else None
        print(
            f"[{style}] rank={rank!s:>4}  nDCG@10={m.get('nDCG@10', 0):.3f}  "
            f"RR={m.get('RR', 0):.3f}  q={query!r}"
        )


def _load_examples() -> list:
    client = Client()
    print(f"Fetching examples from LangSmith dataset '{RETRIEVAL_DATASET}'...")
    examples = list(client.list_examples(dataset_name=RETRIEVAL_DATASET))
    if not examples:
        raise SystemExit(f"No examples found in dataset '{RETRIEVAL_DATASET}'.")
    print(f"Loaded {len(examples)} examples.")
    return examples


@app.command()
def evaluate(
    bm25: Annotated[
        float,
        Option("--bm25", min=0.0, max=1.0, help="BM25 / lexical weight."),
    ] = DEFAULT_SEARCH_PARAMS.bm25_weight,
    vector: Annotated[
        float,
        Option("--vector", min=0.0, max=1.0, help="Vector semantic weight."),
    ] = DEFAULT_SEARCH_PARAMS.vector_weight,
    threshold: Annotated[
        float,
        Option("--threshold", min=0.0, max=1.0, help="Cosine distance threshold."),
    ] = DEFAULT_SEARCH_PARAMS.cosine_distance_threshold,
    limit: Annotated[
        int,
        Option("--limit", min=1, help="Maximum candidates to return."),
    ] = DEFAULT_SEARCH_PARAMS.result_limit,
    per_query: Annotated[
        bool,
        Option("--per-query/--no-per-query", help="Print per-query rank / metrics."),
    ] = False,
    as_json: Annotated[
        bool,
        Option("--json/--no-json", help="Dump full metrics report as JSON."),
    ] = False,
) -> None:
    """Run a single offline evaluation with the given search parameters."""
    params = SearchParams(
        bm25_weight=bm25,
        vector_weight=vector,
        cosine_distance_threshold=threshold,
        result_limit=limit,
    )

    async def _run() -> None:
        examples = _load_examples()
        queries = [ex.inputs["query"] for ex in examples]
        print("Pre-computing query embeddings...")
        embeddings = await precompute_query_embeddings(queries)
        print(f"Evaluating with {params.as_dict()} ...")
        report = await _evaluate_once(examples, embeddings, params)
        print(format_metrics(report))
        print(f"\nparams: {json.dumps(report.get('params', {}), indent=2)}")
        if per_query:
            rankings = await _collect_rankings(examples, embeddings, params)
            _print_per_query(examples, rankings)
        if as_json:
            print(json.dumps(report, indent=2, default=str))

    asyncio.run(_run())


@app.command()
def optimize(
    n_trials: Annotated[
        int,
        Option("--n-trials", min=1, help="Number of Optuna trials."),
    ] = 100,
    limit: Annotated[
        int,
        Option("--limit", min=1, help="Maximum candidates per search call."),
    ] = DEFAULT_SEARCH_PARAMS.result_limit,
) -> None:
    """Find optimal retrieval hyperparameters via Optuna (TPE sampler)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    async def _objective_async(trial: optuna.Trial) -> float:
        bm25_w = trial.suggest_float("bm25_weight", 0.0, 1.0)
        vec_w = round(1.0 - bm25_w, 4)
        thr = trial.suggest_float("cosine_distance_threshold", 0.3, 0.9)
        params = SearchParams(
            bm25_weight=bm25_w,
            vector_weight=vec_w,
            cosine_distance_threshold=thr,
            result_limit=limit,
        )
        examples = _load_examples()
        queries = [ex.inputs["query"] for ex in examples]
        embeddings = await precompute_query_embeddings(queries)

        # Evaluate in 4 progressive chunks for early stopping
        chunks = [examples[i::4] for i in range(4)]
        running_ndcg: list[float] = []
        for i, chunk in enumerate(chunks):
            report = await _evaluate_once(chunk, embeddings, params)
            ndcg = float(report["overall"].get("nDCG@10", 0.0))
            running_ndcg.append(ndcg)
            trial.report(sum(running_ndcg) / len(running_ndcg), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(running_ndcg) / len(running_ndcg)

    with asyncio.Runner() as runner:

        def objective(trial: optuna.Trial) -> float:
            return runner.run(_objective_async(trial))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=4, reduction_factor=3
            ),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_val = study.best_value
    print("OPTIMAL HYPER-PARAMETERS")
    print(f"bm25_weight: {best['bm25_weight']:.4f}")
    print(f"vector_weight: {round(1.0 - best['bm25_weight'], 4):.4f}")
    print(f"cosine_distance_threshold: {best['cosine_distance_threshold']:.4f}")
    print(f"Best nDCG@10: {best_val:.4f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
