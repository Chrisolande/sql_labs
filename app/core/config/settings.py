from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    env: str = "development"
    database_url: str | None = None
    google_api_key: str | None = None
    deepseek_api_key: str | None = None
    nvidia_api_key: str | None = None
    job_data_lake_api_key: str | None = None
    jina_api_key: str | None = None
    tavily_api_key: str | None = None
    discovery_page_cap: int | None = 1
    discovery_per_page: int = 1
    discovery_unconfirmed_limit: int = 3

    # Fallback configuration
    fallback_enabled: bool = True
    fallback_min_result_threshold: int = 5
    fallback_max_results: int = 20

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        if self.env == "production":
            default_db_url = "postgresql+asyncpg://localhost:5432/project"
            if self.database_url == default_db_url:
                raise ValueError(
                    "database_url must be explicitly configured in production"
                )
        return self

    @model_validator(mode="after")
    def validate_langsmith(self) -> "Settings":
        if self.langsmith_tracing:
            missing = [
                name
                for name, value in (
                    ("LANGSMITH_API_KEY", self.langsmith_api_key),
                    ("LANGSMITH_PROJECT", self.langsmith_project),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"LANGSMITH_TRACING is true but missing: {', '.join(missing)}"
                )
        return self


settings = Settings()
