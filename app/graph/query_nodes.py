from typing import Any

from langgraph.types import Command, Send

from app.core.db.base import async_session
from app.core.jdl.schemas import JobSearchCriteria
from app.graph.graph_state import QAGraphState
from app.graph.models import get_fast_model
from app.retrieval.hybrid_search import search_jobs

MAX_HEURISTIC_ATTEMPTS = 2
DEFAULT_TOP_N = 5


async def load_career_memories_from_store(
    store: Any, user_id: str, limit_per_type: int = 3
) -> str:
    if not store or not user_id:
        return ""
    snippets: list[str] = []
    entity_types = [
        "skill",
        "project",
        "achievement",
        "employment_history",
        "education",
    ]
    for et in entity_types:
        try:
            if hasattr(store, "asearch"):
                results = await store.asearch((user_id, et), limit=limit_per_type)
                for r in results:
                    val = (
                        r.value if isinstance(r.value, dict) else {"text": str(r.value)}
                    )
                    txt = val.get("text") or val.get("content") or str(val)[:120]
                    snippets.append(f"{et}:{txt}")
            elif hasattr(store, "aget"):
                item = await store.aget((user_id, et), "default")
                if item:
                    snippets.append(f"{et}:{str(item.value)[:120]}")
        except Exception:
            continue
    if snippets:
        return (
            "\nUser career memory (from Postgres Store, migrated from career_memory table): "
            + "; ".join(snippets[:6])
        )
    return ""


async def migrate_career_memory_table_to_store(
    db_session, store: Any, user_id: str
) -> int:
    """
    One-time migration helper: copies rows from career_memory table into Postgres Store.
    """
    if not store:
        return 0
    try:
        import uuid

        from sqlalchemy import select

        from app.core.db.models.memory import CareerMemory

        uid = uuid.UUID(user_id)
        q = await db_session.execute(
            select(CareerMemory)
            .where(CareerMemory.user_id == uid, CareerMemory.valid_to.is_(None))
            .limit(50)
        )
        rows = q.scalars().all()
        count = 0
        for row in rows:
            ns = (
                user_id,
                row.entity_type.value
                if hasattr(row.entity_type, "value")
                else str(row.entity_type),
            )
            key = row.fact_key or "default"
            value = (
                row.content
                if isinstance(row.content, dict)
                else {"text": str(row.content)}
            )
            try:
                if hasattr(store, "aput"):
                    await store.aput(ns, key, value)
                    count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


async def analyze_query(state: QAGraphState, *, store: Any | None = None):
    user_snippet = ""
    if store and state.user_id:
        try:
            user_snippet = await load_career_memories_from_store(store, state.user_id)
        except Exception:
            user_snippet = ""

    model = get_fast_model().with_structured_output(JobSearchCriteria)
    prompt = (
        "Extract job search criteria from query. Only include fields explicitly mentioned, never guess.\n"
        f"{user_snippet}\n\nQuery: {state.user_query}"
    )
    criteria = await model.ainvoke(prompt)
    return {"extracted_criteria": criteria}


def build_query_text(state: QAGraphState) -> str:
    parts = [state.user_query]
    c = state.extracted_criteria
    if c:
        for field, is_list in [
            ("skills", True),
            ("job_function", False),
            ("seniority", True),
            ("location", False),
            ("remote_type", False),
        ]:
            val = getattr(c, field, None)
            if val:
                parts.append(" ".join(val) if is_list else val)
    return " ".join(parts)


async def hybrid_search(state: QAGraphState) -> dict:
    query_text = build_query_text(state)
    limit = state.extracted_criteria.limit if state.extracted_criteria else None
    async with async_session() as db:
        results = await search_jobs(
            db=db, query_text=query_text, result_limit=limit or 20
        )
    return {"retrieved_jobs": results}


def route_to_scoring(state: QAGraphState):
    if not state.retrieved_jobs:
        return "generate_draft"
    return [
        Send("score_candidate", {"user_query": state.user_query, "job_to_score": job})
        for job in state.retrieved_jobs
    ]


def detect_submit_intent(state: QAGraphState) -> bool:
    q = (state.user_query or "").lower()
    return any(k in q for k in ["submit", "apply", "send application"])


async def intent_router(state: QAGraphState) -> Command:
    if detect_submit_intent(state):
        return Command(goto="prepare_submission")
    return Command(goto="hybrid_search")
