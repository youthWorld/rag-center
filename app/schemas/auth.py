from pydantic import BaseModel


class AuthMeFeatures(BaseModel):
    allowed_profiles: list[str]
    hybrid_allowed: bool
    rerank_allowed: bool
    query_rewrite_allowed: bool
    evidence_allowed: bool
    research_allowed: bool


class AuthMeLimits(BaseModel):
    retrieve_qps: int
    retrieve_daily: int
    max_kb: int
    max_kb_per_retrieve: int
    max_documents_per_kb: int
    max_processing_documents: int


class AuthMeUsage(BaseModel):
    kb_count: int
    retrieve_daily_count: int


class AuthMeResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    key_prefix: str | None = None
    key_name: str | None = None
    plan: str
    features: AuthMeFeatures
    limits: AuthMeLimits
    usage: AuthMeUsage
