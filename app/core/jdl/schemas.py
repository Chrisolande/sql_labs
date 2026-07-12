import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class JobSearchCriteria(BaseModel):
    model_config = {"populate_by_name": True}
    skills: list[str] | None = Field(
        default=None,
        alias="q",
        description="Specific skills or keywords to search for. ONLY include explicitly mentioned terms.",
    )
    location: str | None = Field(
        default=None,
        description="Location mentioned by the user. ONLY include if explicitly mentioned.",
    )
    remote_type: str | None = Field(
        default=None,
        description="Remote preference (e.g., 'remote', 'onsite'). ONLY include if explicitly mentioned.",
    )
    job_function: str | None = Field(
        default=None,
        description="Job title or function (e.g., 'Software Engineer'). ONLY include if explicitly mentioned.",
    )
    seniority: list[str] | None = Field(
        default=None,
        description="Seniority levels (e.g., 'Senior', 'Staff'). CRITICAL: ONLY include levels explicitly requested by the user. NEVER hallucinate or enumerate possible levels.",
    )
    limit: int | None = Field(
        default=None,
        description="The specific number of jobs/positions requested by the user, if explicitly stated (e.g. 'find 4 jobs' -> 4).",
    )


class JobBase(BaseModel):
    title: str
    company_name: str
    domain_name: str | None = None
    role: str | None = None
    job_function: str | None = None
    seniority: list[str] = Field(default_factory=list)
    employment_type: str | None = None
    remote_type: str | None = None
    locations: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    employee_count: str | None = None
    funding: str | None = None
    posted_at: datetime | None = None


class JobRead(JobBase):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    dedup_hash: str
    status: str = "active"


class JobCreate(JobBase):
    dedup_hash: str
    status: str = "active"
    source: "JobSourceCreate"


class JobSearchResult(JobRead):
    company_summary: str | None = None
    rrf_score: float


class JobSourceCreate(BaseModel):
    source_name: str
    source_job_id: str
    source_url: str
    first_seen_at: datetime


class JobCandidate(BaseModel):
    job: JobRead
    score: float


class JobEmbeddingDocument(BaseModel):
    title: str | None = None
    company: str | None = None
    role: str | None = None
    function: str | None = None
    seniority: list[str] = []
    employment: str | None = None
    remote: str | None = None
    locations: list[str] = []
    skills: list[str] = []
    description: str | None = None
    company_summary: str | None = None
