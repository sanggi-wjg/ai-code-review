from typing import Dict, List

import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from starlette import status
from starlette.responses import StreamingResponse, Response

from app.dto.request_dto import CodeReviewRequestDto, CodeChatRequestDto, RepositoryIndexRequestDto
from app.service import CodeReviewService, CodeChatService

app = FastAPI()


@app.get("/assistants/models", status_code=status.HTTP_200_OK)
async def get_models() -> List[str]:
    response = requests.get("http://localhost:11434/api/tags")
    response.raise_for_status()
    return [r["name"] for r in response.json()["models"]]


@app.post("/assistant/review", status_code=status.HTTP_202_ACCEPTED)
async def request_code_review(
    request_dto: CodeReviewRequestDto,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    background_tasks.add_task(
        CodeReviewService.review,
        request_dto.model,
        request_dto.github_token,
        request_dto.repository,
        request_dto.pr_number,
    )
    return {"message": "Request of code review are accepted"}


@app.put("/assistant/repositories/index", status_code=status.HTTP_202_ACCEPTED)
async def index_repository(
    request_dto: RepositoryIndexRequestDto,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    background_tasks.add_task(
        CodeChatService.index,
        request_dto.repository,
        request_dto.language,
    )
    return {"message": "Request of code index are accepted"}


@app.post("/assistant/repositories/chat", status_code=status.HTTP_200_OK)
async def repositories_code_chat(request_dto: CodeChatRequestDto) -> dict:
    return CodeChatService.chat_about_repository(
        request_dto.repository,
        request_dto.search,
    )


@app.post("/assistant/repositories/generate", status_code=status.HTTP_200_OK)
async def repositories_code_regenerate(request_dto: CodeChatRequestDto) -> StreamingResponse:
    return StreamingResponse(
        CodeChatService.chat_to_generate_code(
            request_dto.code,
            request_dto.repository,
            request_dto.language,
            request_dto.search,
            request_dto.consideration,
        ),
        status_code=status.HTTP_200_OK,
        media_type="text/event-stream",
    )


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        port=8000,
        reload=False,
        reload_delay=1,
        use_colors=True,
    )
