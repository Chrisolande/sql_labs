from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from httpx_retries import Retry, RetryTransport

from app.core.config.settings import settings
from app.core.jdl.schemas import JobSearchCriteria
from app.core.pagination import paginate_api

BASE_URL = "https://api.jobdatalake.com/v1"


def build_search_params(criteria: JobSearchCriteria) -> dict[str, Any]:
    params: dict[str, Any] = {}

    q_parts = []
    if criteria.job_function:
        q_parts.append(criteria.job_function)
    if criteria.skills:
        q_parts.extend(criteria.skills)

    if q_parts:
        params["q"] = " ".join(q_parts)

    if criteria.location:
        params["location"] = criteria.location
    if criteria.remote_type:
        params["remote_type"] = criteria.remote_type
    return params


@dataclass
class JobDataLakeClient:
    api_key: str | None = None
    timeout: float = 30.0
    limiter: AsyncLimiter = field(default_factory=lambda: AsyncLimiter(30, 60))
    retry: Retry = field(
        default_factory=lambda: Retry(
            total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )
    )
    client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "API key required. Set JOB_DATA_LAKE_API_KEY environment "
                "variable to use the API."
            )

    async def __aenter__(self):
        self.client = httpx.AsyncClient(
            headers={"X-Api-Key": self.api_key, "Accept": "application/json"},
            timeout=self.timeout,
            transport=RetryTransport(retry=self.retry),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def search_jobs(self, per_page: int = 50, **extra_params: Any) -> dict:
        if not self.client:
            raise RuntimeError("Client not initialized...")
        url = f"{BASE_URL}/jobs"
        params: dict[str, Any] = {"per_page": per_page, **extra_params}
        async with self.limiter:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def search_all_results(
        self,
        criteria: JobSearchCriteria,
        max_results: int | None = None,
        per_page: int = 50,
        page_cap: int | None = None,
    ) -> AsyncIterator[dict]:
        page_cap = page_cap or settings.discovery_page_cap
        search_params = build_search_params(criteria)

        async def _fetch_page(page: int, per_page: int) -> dict:
            return await self.search_jobs(per_page=per_page, page=page, **search_params)

        return paginate_api(
            _fetch_page, lambda d: d.get("jobs", []), per_page, max_results, page_cap
        )

    async def get_job_by_id(self, external_id: str) -> dict | None:
        if not self.client:
            raise RuntimeError("Client not initialized.")
        url = f"{BASE_URL}/jobs/{external_id}"
        async with self.limiter:
            response = await self.client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
