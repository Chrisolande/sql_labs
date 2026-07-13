from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.jdl.schemas import JobSearchResult
from app.core.llm.embeddings import get_embeddings_client


async def search_jobs(
    db: AsyncSession,
    query_text: str,
    cosine_distance_threshold: float = 0.4513,
    bm25_weight: float = 0.2069,
    vector_weight: float = 0.7931,
    result_limit: int = 20,
    candidate_limit: int = 150,
) -> list[JobSearchResult]:
    embeddings_client = get_embeddings_client(task_type="retrieval_query")
    query_embedding = await embeddings_client.aembed_query(query_text)

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
            "cosine_distance_threshold": cosine_distance_threshold,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "result_limit": result_limit,
            "candidate_limit": candidate_limit,
        },
    )
    rows = result.mappings().all()
    return [JobSearchResult.model_validate(dict(row)) for row in rows]
