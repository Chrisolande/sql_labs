from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

from app.core.config.settings import settings
from app.graph.graph_state import QAGraphState
from app.graph.query_nodes import (
    analyze_query,
    hybrid_search,
    intent_router,
    route_to_scoring,
)
from app.graph.scoring_nodes import (
    compile_results,
    generate_draft,
    heuristic_check,
    llm_critic,
    score_candidate,
)
from app.graph.submission_nodes import (
    execute_submission,
    prepare_submission,
    request_human_approval,
)


def _build_graph(checkpointer: Any, store: Any):
    """Internal builder, expects Postgres checkpointer and store already setup."""
    graph = StateGraph(QAGraphState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("intent_router", intent_router)
    graph.add_node("hybrid_search", hybrid_search)
    graph.add_node("score_candidate", score_candidate)
    graph.add_node("compile_results", compile_results)
    graph.add_node("prepare_submission", prepare_submission)
    graph.add_node("request_human_approval", request_human_approval)
    graph.add_node("execute_submission", execute_submission)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("heuristic_check", heuristic_check)
    graph.add_node("llm_critic", llm_critic)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "intent_router")
    graph.add_conditional_edges(
        "hybrid_search", route_to_scoring, ["score_candidate", "generate_draft"]
    )
    graph.add_edge("score_candidate", "compile_results")
    graph.add_edge("compile_results", "generate_draft")
    graph.add_edge("generate_draft", "heuristic_check")
    graph.add_edge("prepare_submission", "request_human_approval")
    graph.add_edge("execute_submission", "generate_draft")

    return graph.compile(checkpointer=checkpointer, store=store)


def build_qa_graph(*, checkpointer: Any, store: Any):
    """Build graph with Postgres persistence - rejects in-memory checkpointer."""
    if checkpointer is None or store is None:
        raise ValueError(
            "Postgres checkpointer and Postgres Store are required. Use build_qa_graph_postgres() or pass async instances."
        )

    name = type(checkpointer).__name__.lower()
    if "memory" in name or "inmemory" in name:
        raise ValueError(
            f"Memory checkpointer {name} not allowed. Must be Postgres (AsyncPostgresSaver)."
        )
    return _build_graph(checkpointer, store)


@asynccontextmanager
async def build_qa_graph_postgres(dsn: str | None = None) -> AsyncGenerator[Any]:
    dsn = (
        dsn
        or getattr(settings, "database_url", None)
        or getattr(settings, "postgres_dsn", None)
    )
    if not dsn:
        raise ValueError(
            "DATABASE_URL not set in settings. Set settings.database_url for Postgres checkpointer."
        )

    checkpointer_pool = AsyncConnectionPool(
        conninfo=dsn, max_size=20, kwargs={"autocommit": True, "prepare_threshold": 0}
    )
    store_pool = AsyncConnectionPool(
        conninfo=dsn, max_size=20, kwargs={"autocommit": True, "prepare_threshold": 0}
    )

    await checkpointer_pool.open()
    await store_pool.open()

    checkpointer = AsyncPostgresSaver(checkpointer_pool)
    store = AsyncPostgresStore(store_pool)

    await checkpointer.setup()
    await store.setup()

    try:
        graph = _build_graph(checkpointer=checkpointer, store=store)
        yield graph
    finally:
        await checkpointer_pool.close()
        await store_pool.close()
