import textwrap
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from app.evaluation.constants import MATCHING_STYLES, NO_MATCH_STYLE

__all__ = [
    "DEFAULT_SEARCH_PARAMS",
    "MATCHING_STYLES",
    "NO_MATCH_STYLE",
    "RETRIEVAL_DATASET",
    "SearchParams",
    "precompute_query_embeddings",
    "retrieval_target",
    "run_search",
    "search_jobs_with_embedding",
]

RETRIEVAL_DATASET = "careerpilot-matching-eval-v2"


@dataclass(frozen=True, slots=True)
class SearchParams:
    bm25_weight: float = 0.3302
    vector_weight: float = 0.6698
    cosine_distance_threshold: float = 0.3794
    result_limit: int = 20
    candidate_limit: int = 150

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SEARCH_PARAMS = SearchParams()


async def search_jobs_with_embedding(
    db: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    params: SearchParams = DEFAULT_SEARCH_PARAMS,
) -> list[Any]:
    stmt = text("""
        SELECT * FROM hybrid_search_jobs(
            :query_text, :query_embedding, :cosine_distance_threshold,
            :bm25_weight, :vector_weight, :result_limit, :candidate_limit
        )
    """)
    result = await db.execute(
        stmt,
        {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "cosine_distance_threshold": params.cosine_distance_threshold,
            "bm25_weight": params.bm25_weight,
            "vector_weight": params.vector_weight,
            "result_limit": params.result_limit,
            "candidate_limit": params.candidate_limit,
        },
    )
    return result.mappings().all()


async def run_search(
    query_text: str,
    params: SearchParams = DEFAULT_SEARCH_PARAMS,
    *,
    query_embedding: list[float] | None = None,
    include_snippets: bool = False,
) -> list[dict[str, Any]]:
    from app.core.db.base import async_session  # noqa: PLC0415
    from app.core.db.models.job import JobDescription  # noqa: PLC0415
    from app.core.jdl.schemas import JobSearchResult  # noqa: PLC0415
    from app.retrieval.hybrid_search import search_jobs  # noqa: PLC0415

    async with async_session() as session:
        if query_embedding is not None:
            rows = await search_jobs_with_embedding(
                session, query_text, query_embedding, params
            )
            candidates = [JobSearchResult.model_validate(dict(row)) for row in rows]
        else:
            candidates = await search_jobs(
                session,
                query_text=query_text,
                cosine_distance_threshold=params.cosine_distance_threshold,
                bm25_weight=params.bm25_weight,
                vector_weight=params.vector_weight,
                result_limit=params.result_limit,
                candidate_limit=params.candidate_limit,
            )

        descriptions: dict = {}
        if include_snippets and candidates:
            job_ids = [c.id for c in candidates]
            stmt = select(JobDescription).where(JobDescription.job_id.in_(job_ids))
            result = await session.execute(stmt)
            descriptions = {desc.job_id: desc.cleaned_text for desc in result.scalars()}

    ranked: list[dict[str, Any]] = []
    for c in candidates:
        item = c.model_dump(
            include={"id", "title", "company_name", "required_skills"},
            mode="json",
        )
        item["id"] = str(item["id"])  # ensure string
        item["score"] = float(c.rrf_score)
        item["required_skills"] = item["required_skills"] or []
        if include_snippets:
            desc = descriptions.get(c.id, "") or ""
            item["description_snippet"] = (
                textwrap.shorten(desc, width=500, placeholder="...") if desc else ""
            )
        ranked.append(item)
    return ranked


@retry(wait=wait_exponential_jitter(initial=3, max=60), stop=stop_after_attempt(6))
async def _embed_query(client: Any, query: str) -> list[float]:
    return await client.aembed_query(query)


async def precompute_query_embeddings(queries: list[str]) -> dict[str, list[float]]:
    from app.core.llm.embeddings import get_embeddings_client  # noqa: PLC0415

    client = get_embeddings_client(task_type="retrieval_query")
    unique = list(dict.fromkeys(queries))
    embeddings: dict[str, list[float]] = {}
    for q in unique:
        embeddings[q] = await _embed_query(client, q)
    return embeddings


async def retrieval_target(
    inputs: dict,
    params: SearchParams = DEFAULT_SEARCH_PARAMS,
) -> dict:
    query = inputs.get("query", "")
    ranked_jobs = await run_search(
        query_text=query,
        params=params,
        include_snippets=True,
    )
    return {
        "ranked_jobs": ranked_jobs,
        "ranked_job_ids": [j["id"] for j in ranked_jobs],
        "search_params": params.as_dict(),
    }
