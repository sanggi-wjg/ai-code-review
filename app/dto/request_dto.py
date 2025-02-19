from pydantic import BaseModel, Field


class CodeReviewRequestDto(BaseModel):
    model: str
    github_token: str
    repository: str
    pr_number: int


class RepositoryIndexRequestDto(BaseModel):
    github_token: str
    repository: str
    language: str


class CodeChatRequestDto(BaseModel):
    code: str
    repository: str
    language: str = Field("", description="언어, python, kotlin, ...")
    search: str = Field("", description="검색")
    consideration: str = Field("", description="고려사항")
