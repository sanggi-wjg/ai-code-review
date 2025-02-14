import uvicorn
from fastapi import FastAPI, BackgroundTasks
from starlette import status

from app.dto.review_request_dto import ReviewRequestDto
from app.service import LlmReviewService

app = FastAPI()


@app.post("/assistant/review", status_code=status.HTTP_202_ACCEPTED)
async def review_request(review_request_dto: ReviewRequestDto, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        LlmReviewService.review,
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
        port=8000,
        reload=False,
        use_colors=True,
    )
