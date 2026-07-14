import operator
from typing import Annotated

from pydantic import BaseModel, Field

from app.core.jdl.schemas import JobSearchCriteria, JobSearchResult


class JobClaim(BaseModel):
    text: str = Field(description="The claim being made, in natural language.")
    job_id: str = Field(description="The id of the job this claim is grounded in.")


class JobScore(BaseModel):
    job_id: str = Field(description="The id of the scored job.")
    fit_score: float = Field(description="Fit score 0.0-1.0.", ge=0.0, le=1.0)
    rationale: str = Field(description="One-sentence reason for the score.")


class GeneratedResponse(BaseModel):
    insufficient_data: bool = Field(
        description="True if retrieved jobs lack info to answer."
    )
    summary: str = Field(description="Natural language answer.")
    claims: list[JobClaim] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    is_valid: bool = Field(description="True if draft is faithful to retrieved jobs.")
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = Field(default=None)


class QAGraphState(BaseModel):
    user_query: str
    user_id: str | None = Field(
        default=None,
        description="Tenant-isolated identity, set by transport layer, never by LLM",
    )

    extracted_criteria: JobSearchCriteria | None = None

    retrieved_jobs: list[JobSearchResult] = Field(default_factory=list)

    draft_response: GeneratedResponse | None = None

    heuristic_attempts: int = 0
    verification_attempts: int = 0

    grounding_error: str | None = None
    critique_feedback: str | None = None

    # Submission HIL fields
    application_id: str | None = Field(default=None)
    pending_job_id: str | None = Field(default=None)
    human_approval: bool | None = Field(default=None)
    submission_preview: dict | None = Field(default=None)
    submission_result: dict | None = Field(default=None)

    # Fan-out helpers
    jobs_to_score: JobSearchResult | None = None
    job_scores: Annotated[list[JobScore], operator.add] = Field(default_factory=list)
