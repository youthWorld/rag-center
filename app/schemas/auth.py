from pydantic import BaseModel


class AuthMeResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    key_prefix: str | None = None
    key_name: str | None = None
