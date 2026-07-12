from app.core.jdl.schemas import JobSearchCriteria

DISCOVERY_SEED_CRITERIA: list[JobSearchCriteria] = [
    JobSearchCriteria(job_function="Software Engineer"),
    JobSearchCriteria(job_function="Product Manager"),
    JobSearchCriteria(job_function="Data Scientist"),
    JobSearchCriteria(job_function="Marketing Manager"),
    JobSearchCriteria(job_function="Sales"),
    JobSearchCriteria(skills=["python"]),
    JobSearchCriteria(skills=["react"]),
]
