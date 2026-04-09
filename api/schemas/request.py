from pydantic import BaseModel, field_validator


class AnalyzeRequest(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code must not be empty.")
        return v