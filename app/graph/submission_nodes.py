import re
import uuid
from typing import Any

from langgraph.graph import END
from langgraph.types import Command, interrupt
from sqlalchemy import select

from app.core.db.base import async_session
from app.core.db.models.job import Job
from app.graph.graph_state import GeneratedResponse, QAGraphState


def _resp(summary: str, *, insufficient: bool = True) -> GeneratedResponse:
    return GeneratedResponse(insufficient_data=insufficient, summary=summary, claims=[])


async def prepare_submission(state: QAGraphState, *, store: Any | None = None):
    m = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        state.user_query,
        re.I,
    )
    job_id_str = (
        m.group(0)
        if m
        else (str(state.retrieved_jobs[0].id) if state.retrieved_jobs else None)
    )
    if not job_id_str:
        return {
            "draft_response": _resp(
                "Specify job ID to submit, or search first then say 'submit'."
            ),
            "submission_preview": None,
        }

    preview = {"job_id": job_id_str}
    app_id = None
    try:
        async with async_session() as db:
            job = await db.get(Job, uuid.UUID(job_id_str))
            if job:
                preview.update({"title": job.title, "company": job.company_name})
            if state.user_id:
                from app.core.db.models.application import Application

                q = await db.execute(
                    select(Application)
                    .where(
                        Application.user_id == uuid.UUID(state.user_id),
                        Application.job_id == uuid.UUID(job_id_str),
                        Application.status == "draft",
                    )
                    .limit(1)
                )
                draft = q.scalar_one_or_none()
                if draft:
                    app_id = str(draft.id)
                    preview["resume_present"] = bool(draft.resume_draft)
    except Exception:
        pass
    return {
        "pending_job_id": job_id_str,
        "application_id": app_id,
        "submission_preview": preview,
    }


async def request_human_approval(state: QAGraphState) -> Command:
    if not state.submission_preview or not state.pending_job_id:
        return Command(
            goto=END,
            update={"draft_response": _resp("Missing job or draft for submission.")},
        )
    payload = {
        "action": "submit_application",
        "job_id": state.pending_job_id,
        "application_id": state.application_id,
        "preview": state.submission_preview,
        "message": f"Approve submission for job {state.pending_job_id}?",
    }
    human = interrupt(payload)
    approved = (
        bool(human.get("approved") if isinstance(human, dict) else human)
        if human
        else False
    )
    if approved:
        return Command(goto="execute_submission", update={"human_approval": True})
    return Command(
        goto=END,
        update={
            "draft_response": _resp(
                "Submission cancelled. Draft kept.", insufficient=False
            ),
            "human_approval": False,
        },
    )


async def execute_submission(state: QAGraphState):
    if not state.human_approval:
        return {
            "submission_result": {"status": "cancelled"},
            "draft_response": _resp("Not approved.", insufficient=False),
        }
    try:
        from app.core.db.models.application import Application

        if not state.user_id:
            raise PermissionError("Missing user context")
        async with async_session() as db:
            app_obj = None
            if state.application_id:
                app_obj = await db.get(Application, uuid.UUID(state.application_id))
            if not app_obj and state.pending_job_id:
                app_obj = Application(
                    user_id=uuid.UUID(state.user_id),
                    job_id=uuid.UUID(state.pending_job_id),
                    status="draft",
                )
                db.add(app_obj)
                await db.flush()
            if not app_obj:
                raise ValueError("Application not found")
            if str(app_obj.user_id) != str(state.user_id):
                raise PermissionError("Cross-tenant blocked")
            app_obj.status = "submitted"
            await db.commit()
            await db.refresh(app_obj)
            return {
                "submission_result": {
                    "status": "submitted",
                    "application_id": str(app_obj.id),
                },
                "application_id": str(app_obj.id),
                "draft_response": _resp(
                    f"Application {app_obj.id} submitted for job {state.pending_job_id}.",
                    insufficient=False,
                ),
            }
    except Exception as e:
        return {
            "submission_result": {"status": "failed", "error": str(e)},
            "draft_response": _resp(f"Submission failed: {e}"),
        }
