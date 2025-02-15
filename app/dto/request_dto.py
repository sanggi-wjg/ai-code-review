from pydantic import BaseModel, Field


class CodeReviewRequestDto(BaseModel):
    github_token: str
    groq_api_key: str
    groq_model: str = "deepseek-r1-distill-llama-70b"
    repository: str
    pr_number: int


class RepositoryIndexRequestDto(BaseModel):
    github_token: str
    repository: str
    language: str


class CodeChatRequestDto(BaseModel):
    code: str
    repository: str
    search: str = Field("", description="검색")
    consideration: str = Field("", description="고려사항")
