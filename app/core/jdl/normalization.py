import hashlib
from datetime import UTC, datetime
from typing import Any

from more_itertools import unique_everseen
from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.core.jdl.schemas import JobCreate, JobSourceCreate


def collapse_whitespace(s: str) -> str:
    return " ".join(s.strip().lower().split()) if s else ""


def first_present(raw: dict, *keys: str) -> Any:
    return next((raw.get(k) for k in keys if raw.get(k)), None)


def compute_dedup_hash(company_name: str, title: str, skills: list[str]) -> str:
    norm_company = collapse_whitespace(company_name)
    norm_title = collapse_whitespace(title)
    norm_skills = sorted([collapse_whitespace(s) for s in skills if s])

    data = f"{norm_company}|{norm_title}|{','.join(norm_skills)}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def normalize_skills(skills: list[str] | None) -> list[str]:
    if not skills:
        return []
    stripped = [s.strip() for s in skills if s and s.strip()]
    return list(unique_everseen(stripped, key=str.lower))


class RawJobInput(BaseModel):
    title: str = Field(default="")
    locations: list[str] = Field(default_factory=list)
    company_name: str = Field(
        default="", validation_alias=AliasChoices("company_name", "company")
    )
    domain_name: str | None = None
    role: str | None = None
    job_function: str | None = None
    seniority: list[str] = Field(default_factory=list)
    employment_type: str | None = None
    remote_type: str | None = None
    countries: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    employee_count: str | None = None
    funding: str | None = None

    source_url: str = Field(
        default="",
        validation_alias=AliasChoices("url", "absolute_url", "hostedUrl", "jobUrl"),
    )
    source_job_id: str | int = Field(
        default="", validation_alias=AliasChoices("id", "job_handle")
    )

    # Parse the dates explicitly in validators
    posted_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "posted_at", "updated_at", "createdAt", "publishedAt"
        ),
    )
    first_seen_at: datetime | None = Field(default=None)

    @classmethod
    def _extract_locations_from_raw(cls, raw: Any) -> list[str]:
        if not isinstance(raw, dict):
            return []

        raw_locations = raw.get("locations")
        if isinstance(raw_locations, list):
            return [str(loc) for loc in raw_locations]
        if raw_locations:
            return [str(raw_locations)]

        gh_loc = raw.get("location")
        if isinstance(gh_loc, dict):
            name = gh_loc.get("name")
            return [str(name)] if name else []
        if isinstance(gh_loc, str):
            return [gh_loc]
        return []

    @model_validator(mode="before")
    @classmethod
    def extract_locations_validator(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "locations" not in data:
                data["locations"] = cls._extract_locations_from_raw(data)
            else:
                data["locations"] = cls._extract_locations_from_raw(data)
        return data


def normalize_job(raw: dict, source_name: str = "primary") -> JobCreate:
    parsed_input = RawJobInput.model_validate(raw)

    skills = normalize_skills(parsed_input.required_skills)

    first_seen_at = parsed_input.first_seen_at or datetime.now(UTC)

    locations = parsed_input.locations
    dedup_hash = compute_dedup_hash(
        parsed_input.company_name, parsed_input.title, skills
    )

    source = JobSourceCreate(
        source_name=source_name,
        source_job_id=str(parsed_input.source_job_id),
        source_url=str(parsed_input.source_url),
        first_seen_at=first_seen_at,
    )

    return JobCreate(
        title=parsed_input.title,
        company_name=parsed_input.company_name,
        domain_name=parsed_input.domain_name,
        role=parsed_input.role,
        job_function=parsed_input.job_function,
        seniority=parsed_input.seniority,
        employment_type=parsed_input.employment_type,
        remote_type=parsed_input.remote_type,
        locations=locations,
        countries=parsed_input.countries,
        required_skills=skills,
        employee_count=parsed_input.employee_count,
        funding=parsed_input.funding,
        posted_at=parsed_input.posted_at,
        dedup_hash=dedup_hash,
        status="active",
        source=source,
    )
