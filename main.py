import uvicorn
from fastapi import FastAPI, BackgroundTasks
from starlette import status

from app.dto.review_request_dto import CodeReviewRequestDto, CodeChatRequestDto
from app.service import CodeReviewService, CodeChatService

app = FastAPI()


@app.post("/assistant/review", status_code=status.HTTP_202_ACCEPTED)
async def code_review_request(request_dto: CodeReviewRequestDto, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        CodeReviewService.review,
        request_dto.github_token,
        request_dto.groq_api_key,
        request_dto.groq_model,
        request_dto.repository,
        request_dto.pr_number,
    )
    return {"message": "Request of code review are accepted"}


@app.post("/assistant/code/chat", status_code=status.HTTP_200_OK)
async def chat(request_dto: CodeChatRequestDto):
    return CodeChatService.chat(request_dto.code)


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=8000,
        reload=False,
        use_colors=True,
    )
