from pydantic import BaseModel


class ReviewRequestDto(BaseModel):
    github_token: str
    groq_api_key: str
    groq_model: str = "deepseek-r1-distill-llama-70b"
    repository: str
    pr_number: int
