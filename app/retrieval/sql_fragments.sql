CREATE OR REPLACE FUNCTION hybrid_search_jobs(
    query_text TEXT,
    query_embedding VECTOR(1024),
    cosine_distance_threshold FLOAT DEFAULT 0.5,
    bm25_weight FLOAT DEFAULT 0.1,
    vector_weight FLOAT DEFAULT 0.9,
    result_limit INT DEFAULT 20,
    candidate_limit INT DEFAULT 150
)
    RETURNS TABLE
            (
                id              UUID,
                dedup_hash      TEXT,
                title           TEXT,
                company_name    TEXT,
                domain_name     TEXT,
                role            TEXT,
                job_function    TEXT,
                seniority       TEXT[],
                employment_type TEXT,
                remote_type     TEXT,
                locations       TEXT[],
                countries       TEXT[],
                required_skills TEXT[],
                employee_count  TEXT,
                funding         TEXT,
                company_summary TEXT,
                status          TEXT,
                posted_at       TIMESTAMPTZ,
                last_seen_at    TIMESTAMPTZ,
                created_at      TIMESTAMPTZ,
                rrf_score       FLOAT
            )
    LANGUAGE sql
    STABLE
AS
$$
WITH bm25_results AS (
    SELECT j.id  AS job_id,
           ROW_NUMBER() OVER (
               ORDER BY j.search_text <@> to_bm25query(query_text, 'idx_jobs_search_bm25') ASC
           ) AS rank
    FROM jobs j
    WHERE j.status = 'active'
      AND j.search_text IS NOT NULL
    ORDER BY j.search_text <@> to_bm25query(query_text, 'idx_jobs_search_bm25') ASC
    LIMIT candidate_limit
),
vector_results AS (
    SELECT e.entity_id                                               AS job_id,
           ROW_NUMBER() OVER (ORDER BY e.vector <=> query_embedding) AS rank
    FROM embeddings e
             JOIN jobs j ON j.id = e.entity_id AND j.status = 'active'
    WHERE e.entity_type = 'job'
      AND (e.vector <=> query_embedding) < cosine_distance_threshold
    ORDER BY e.vector <=> query_embedding
    LIMIT candidate_limit
),
fused AS (
    SELECT COALESCE(b.job_id, v.job_id)                       AS job_id,
           COALESCE(bm25_weight * (1.0 / (60 + b.rank)), 0) +
           COALESCE(vector_weight * (1.0 / (60 + v.rank)), 0) AS rrf_score
    FROM bm25_results b
             FULL OUTER JOIN vector_results v ON b.job_id = v.job_id
)
SELECT j.id,
       j.dedup_hash,
       j.title,
       j.company_name,
       j.domain_name,
       j.role,
       j.job_function,
       j.seniority,
       j.employment_type,
       j.remote_type,
       j.locations,
       j.countries,
       j.required_skills,
       j.employee_count,
       j.funding,
       j.company_summary,
       j.status,
       j.posted_at,
       j.last_seen_at,
       j.created_at,
       f.rrf_score
FROM fused f
         JOIN jobs j ON j.id = f.job_id
WHERE j.status = 'active'
ORDER BY f.rrf_score DESC
LIMIT result_limit;
$$;
