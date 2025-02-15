from typing import Dict

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from starlette import status
from starlette.responses import StreamingResponse

from app.dto.request_dto import CodeReviewRequestDto, CodeChatRequestDto, RepositoryIndexRequestDto
from app.service import CodeReviewService, CodeChatService, VectorStoreService

app = FastAPI()


@app.post("/assistant/review", status_code=status.HTTP_202_ACCEPTED)
async def code_review_request(
    request_dto: CodeReviewRequestDto,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    background_tasks.add_task(
        CodeReviewService.review,
        request_dto.github_token,
        request_dto.groq_api_key,
        request_dto.groq_model,
        request_dto.repository,
        request_dto.pr_number,
    )
    return {"message": "Request of code review are accepted"}


@app.put("/assistant/repositories/index", status_code=status.HTTP_202_ACCEPTED)
async def repositories_index(
    request_dto: RepositoryIndexRequestDto,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    background_tasks.add_task(
        VectorStoreService.index,
        request_dto.repository,
        request_dto.language,
    )
    return {"message": "Request of code index are accepted"}


@app.post("/assistant/repositories/chat", status_code=status.HTTP_200_OK)
async def repositories_code_chat(request_dto: CodeChatRequestDto) -> StreamingResponse:
    return StreamingResponse(
        CodeChatService.chat(
            request_dto.code,
            request_dto.repository,
            request_dto.search,
            request_dto.consideration,
        ),
        # LlmAPI.chat_to_coding_assist_stream(request_dto.code),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
    )


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=8000,
        reload=True,
        reload_delay=1,
        use_colors=True,
    )
