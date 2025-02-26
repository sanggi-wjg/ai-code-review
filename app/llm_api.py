import logging
import traceback
from typing import List, Optional, Generator, Union, Tuple

from langchain.globals import set_debug
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeReviewIssue(BaseModel):
    """Represents a specific issue found during code review."""

    category: str = Field(
        description="Issue category (naming/security/performance)",
        enum=["naming", "security", "performance"],
    )
    description: str = Field(
        description="Detailed description of the identified issue",
    )
    suggestion: str = Field(
        description="Concrete code suggestion for improvement, including code examples where applicable"
    )
    severity: str = Field(
        description="Issue severity level",
        enum=["low", "medium", "high"],
    )


class CodeReviewResult(BaseModel):
    """Represents the result of code review."""

    summary: str = Field(
        description="Overall summary of the code review in Korean, with detailed explanation",
        default="",
    )
    issues: List[CodeReviewIssue] = Field(
        description="List of identified issues with details",
        default_factory=list,
    )
    has_issues: bool = Field(
        description="Indicates whether any critical issues were found in the code",
        default=False,
    )
    review_status: str = Field(
        description="Overall review status",
        enum=["passed", "needs_changes", "critical_issues"],
        default="passed",
    )

    def format_to_comment(self) -> str:
        def get_severity_emoji(severity: str) -> str:
            return {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(severity, "❓")

        def get_status_header(status: str) -> str:
            return {
                "passed": "# ✅ 코드 리뷰 완료",
                "needs_changes": "# ⚠️ 수정이 필요한 사항이 있습니다",
                "critical_issues": "# 🚨 중요한 문제가 발견 되었습니다 🚨",
            }.get(status, "# 코드 리뷰 결과")

        def get_issue_category_title(category_type: str) -> str:
            return {
                "naming": "## 📝 네이밍 이슈",
                "security": "## 🔒 보안 이슈",
                "performance": "## ⚡ 성능 이슈",
            }.get(category_type, "## 🐜 이슈")

        comment = f"{get_status_header(self.review_status)}\n\n"
        comment += f"{self.summary}\n\n"

        if not self.has_issues:
            return comment

        comment += "# 발견된 이슈\n\n"
        for issue in self.issues:
            comment += f"{get_issue_category_title(issue.category)}\n\n"
            comment += f"### {get_severity_emoji(issue.severity)} **문제점**\n"
            comment += f"{issue.description}\n\n"
            comment += f"### 💡 **개선 제안**\n"
            comment += f"{issue.suggestion}\n\n"

        return comment


class LlmAPI:

    SYSTEM_MESSAGE_CODE_REVIEW = """
You are a meticulous and highly skilled code reviewer. Your task is to analyze the following code changes and provide feedback if you identify any of the following critical issues:
- Inappropriate function or variable names that are misleading, unclear, or violate naming conventions, making the code harder to understand or maintain.
- Security vulnerabilities, such as injection risks, improper authentication, data leaks, or insecure dependencies.
- Performance bottlenecks, including inefficient algorithms, redundant computations, memory leaks, or unnecessary resource usage.
        
Summarize your answers to make them as readable as possible. If you identify any issues, please also provide improvements or alternative solutions to address the problem, ensuring that your feedback is actionable and leads to a better solution.
Always respond in Korean. Do not use any other language unless explicitly asked.""".strip()

    SYSTEM_MESSAGE_CODING_ASSIST = """
You are a highly skilled software engineer specializing in developing secure and high-performance backend systems. Your goal is to generate optimized, well-structured, and maintainable code.
Always respond in Korean. Do not use any other language unless explicitly asked.""".strip()

    @classmethod
    def chat_to_review_code(
        cls,
        model: str,
        changes: str,
    ) -> Optional[CodeReviewResult]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", cls.SYSTEM_MESSAGE_CODE_REVIEW),
                ("human", "{changes}"),
            ]
        )

        try:
            logger.info("😢 model not applied to chat_to_review_code.")
            llm = ChatOllama(model="qwen2.5:7b").with_structured_output(CodeReviewResult)
            chain = prompt | llm
            chat_response = chain.invoke({"changes": changes})
            return chat_response
            # for token in chain.stream({"changes": changes}):
            #     yield token
        except:
            traceback.print_exc()
            return None

    @classmethod
    def chat_to_ask(cls, documents: List[Tuple[Document, float]]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a professional software engineer and an expert in code analysis and summarization.
You can quickly and accurately understand a given code and clearly summarize its core functions and behavior.
Always respond in Korean.
""".strip(),
                ),
                ("human", "Summarize the following code.\n{context}"),
            ]
        )
        llm = ChatOllama(model="exaone3.5:7.8b")
        # qa_chain = create_stuff_documents_chain(llm, prompt)
        # chain = create_retrieval_chain(retriever, qa_chain)
        # chat_response = chain.invoke({"input": search})
        context = ""
        for document, score in documents:
            with open(document.metadata["source"], "r") as f:
                context += f.read()

        chain = prompt | llm | StrOutputParser()
        chat_response = ""
        for token in chain.stream({"context": context}):
            print(token, end="", flush=True)
            chat_response += token

        logger.debug(chat_response)
        return chat_response

    @classmethod
    def chat_to_generate_code_stream(
        cls,
        documents: List[Document],
        code: str,
        consideration: str,
    ) -> Generator[str, None, None]:
        user_message = """
Please generate awesome code. I want this code:{code}.

# PROJECT SOURCE CODE SEARCH: 
{documents}
        
# CONSIDERATION: 
{consideration}""".strip()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", cls.SYSTEM_MESSAGE_CODING_ASSIST),
                ("human", user_message),
            ]
        )
        # llm = ChatOllama(model="deepseek-coder-v2:16b")
        llm = ChatOllama(model="qwen2.5-coder:14b")
        chain = prompt | llm | StrOutputParser()

        for token in chain.stream(
            {
                "consideration": consideration,
                "documents": [doc.page_content for doc in documents],
                "code": code,
            }
        ):
            yield token
