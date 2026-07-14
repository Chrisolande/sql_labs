from langgraph.graph import END
from langgraph.types import Command

from app.graph.graph_state import (
    CritiqueResult,
    GeneratedResponse,
    JobScore,
    QAGraphState,
)
from app.graph.models import get_fast_model, get_frontier_model

MAX_HEURISTIC_ATTEMPTS = 2
MAX_VERIFICATION_ATTEMPTS = 2
DEFAULT_TOP_N = 5


def _insufficient(summary: str) -> GeneratedResponse:
    return GeneratedResponse(insufficient_data=True, summary=summary, claims=[])


async def score_candidate(state: dict) -> dict:
    job = state["job_to_score"]
    model = get_fast_model().with_structured_output(JobScore)
    score = await model.ainvoke(
        f"Rate fit 0.0-1.0 with one-sentence rationale.\nUser: {state['user_query']}\n"
        f"Job {job.id}: {job.title} at {job.company_name} skills={job.required_skills} remote={job.remote_type} loc={job.locations}"
    )
    score.job_id = str(job.id)
    return {"job_scores": [score]}


def compile_results(state: QAGraphState) -> dict:
    limit = (
        state.extracted_criteria.limit
        if state.extracted_criteria and state.extracted_criteria.limit
        else DEFAULT_TOP_N
    )
    jobs_by_id = {str(j.id): j for j in state.retrieved_jobs}
    ranked = sorted(state.job_scores, key=lambda s: s.fit_score, reverse=True)
    top = [jobs_by_id[s.job_id] for s in ranked[:limit] if s.job_id in jobs_by_id]
    return {"retrieved_jobs": top, "job_scores": []}


async def generate_draft(state: QAGraphState):
    if not state.retrieved_jobs:
        return {"draft_response": _insufficient("No matching jobs found.")}
    ctx = "\n".join(
        f"job_id={j.id} | {j.title} at {j.company_name} | skills={j.required_skills} | remote={j.remote_type} | loc={j.locations}"
        for j in state.retrieved_jobs
    )
    prompt = f"Answer using ONLY jobs below. Every claim must cite exact job_id. If jobs don't answer query, set insufficient_data=True.\nUser query: {state.user_query}\nRetrieved:\n{ctx}"
    if state.grounding_error:
        prompt += f"\n\nPrevious answer cited invalid job_ids ({state.grounding_error}). Only cite IDs in retrieved list."
    if state.critique_feedback:
        prompt += f"\n\nCritic feedback: {state.critique_feedback}. Fix to be faithful."
    resp = (
        await get_frontier_model()
        .with_structured_output(GeneratedResponse)
        .ainvoke(prompt)
    )
    return {"draft_response": resp, "grounding_error": None, "critique_feedback": None}


def find_ungrounded_claims(state: QAGraphState) -> list[str]:
    if not state.draft_response:
        return []
    valid = {str(j.id) for j in state.retrieved_jobs}
    return [c.job_id for c in state.draft_response.claims if c.job_id not in valid]


async def heuristic_check(state: QAGraphState) -> Command:
    if state.draft_response and state.draft_response.insufficient_data:
        return Command(goto=END)
    bad = find_ungrounded_claims(state)
    if not bad:
        return Command(goto="llm_critic")
    attempts = state.heuristic_attempts + 1
    if attempts >= MAX_HEURISTIC_ATTEMPTS:
        return Command(
            goto=END,
            update={
                "draft_response": _insufficient("Couldn't produce grounded answer."),
                "heuristic_attempts": attempts,
            },
        )
    return Command(
        goto="generate_draft",
        update={"heuristic_attempts": attempts, "grounding_error": ", ".join(bad)},
    )


async def llm_critic(state: QAGraphState) -> Command:
    if not state.draft_response or state.draft_response.insufficient_data:
        return Command(goto=END)
    ctx = "\n".join(
        f"{j.id} | {j.title} at {j.company_name} | {j.required_skills}"
        for j in state.retrieved_jobs
    )
    prompt = f"Strict faithfulness critic. Check draft against retrieved jobs. Flag invented salaries, locations, skills.\nUser: {state.user_query}\nJobs:\n{ctx}\nDraft: {state.draft_response.summary}\nClaims: {[f'{c.job_id}:{c.text}' for c in state.draft_response.claims]}"
    try:
        crit = (
            await get_frontier_model()
            .with_structured_output(CritiqueResult)
            .ainvoke(prompt)
        )
    except Exception:
        return Command(goto=END)
    if crit.is_valid:
        return Command(
            goto=END, update={"verification_attempts": 0, "critique_feedback": None}
        )
    attempts = state.verification_attempts + 1
    if attempts >= MAX_VERIFICATION_ATTEMPTS:
        return Command(
            goto=END,
            update={
                "draft_response": _insufficient("Failed verification after retries."),
                "verification_attempts": attempts,
            },
        )
    fb = "; ".join(crit.issues) if crit.issues else "Semantic mismatch"
    if crit.suggested_fix:
        fb += f" | Fix: {crit.suggested_fix}"
    return Command(
        goto="generate_draft",
        update={"verification_attempts": attempts, "critique_feedback": fb},
    )
