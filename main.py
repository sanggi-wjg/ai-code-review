import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette import status

from app.service import ReviewService

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


app = FastAPI()


class ReviewRequestDto(BaseModel):
    github_token: str
    groq_api_key: str
    groq_model: str = "deepseek-r1-distill-llama-70b"
    repository: str
    pr_number: int


@app.post("/assistant/review", status_code=status.HTTP_202_ACCEPTED)
async def request_review(review_request_dto: ReviewRequestDto, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        ReviewService.review,
        review_request_dto.github_token,
        review_request_dto.groq_api_key,
        review_request_dto.groq_model,
        review_request_dto.repository,
        review_request_dto.pr_number,
    )
    return {"message": "Request of code review are accepted"}


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        reload=False,
        use_colors=True,
    )
    # main("FitpetKorea/fitpetmall-backend", 4007, "deepseek-r1:14b")
    # main("FitpetKorea/fitpetmall-backend-v4", 2658, "deepseek-r1-distill-llama-70b")
