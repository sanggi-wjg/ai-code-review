from typing import List, Optional

from colorful_print import color
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class CodeReviewIssue(BaseModel):
    """Represents a specific issue found during code review."""

    category: str = Field(
        description="Issue category (naming/security/performance)", enum=["naming", "security", "performance"]
    )
    description: str = Field(description="Detailed description of the identified issue")
    suggestion: str = Field(
        description="Concrete code suggestion for improvement, including code examples where applicable"
    )
    severity: str = Field(description="Issue severity level", enum=["low", "medium", "high"])


class CodeReviewResult(BaseModel):
    """Represents the result of code review."""

    summary: str = Field(
        description="Overall summary of the code review in Korean, with detailed explanation", default=""
    )
    issues: List[CodeReviewIssue] = Field(description="List of identified issues with details", default_factory=list)
    has_issues: bool = Field(description="Indicates whether any critical issues were found in the code", default=False)
    review_status: str = Field(
        description="Overall review status", enum=["passed", "needs_changes", "critical_issues"], default="passed"
    )

    def format_for_pr_review_comment(self) -> str:
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

    @classmethod
    def request_code_review(
        cls,
        groq_api_key: str,
        groq_model: str,
        changes: str,
    ) -> Optional[CodeReviewResult]:
        system_message = """
You are a meticulous and highly skilled code reviewer. Your task is to analyze the following code changes and provide feedback **ONLY IF YOU IDENTIFY ANY OF THE FOLLOWING CRITICAL ISSUES**:
- Inappropriate function or variable names that are misleading, unclear, or violate naming conventions, making the code harder to understand or maintain.
- Security vulnerabilities, such as injection risks, improper authentication, data leaks, or insecure dependencies.
- Performance bottlenecks, including inefficient algorithms, redundant computations, memory leaks, or unnecessary resource usage.
        
Summarize your answers to make them as readable as possible. If you identify any issues, please also provide improvements or alternative solutions to address the problem, ensuring that your feedback is actionable and leads to a better solution.
Always respond in Korean. Do not use any other language unless explicitly asked.
""".strip()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{changes}"),
            ]
        )

        # llm = ChatGroq(
        #     model=groq_model,
        #     temperature=0.5,
        #     api_key=groq_api_key,
        #     max_tokens=6000,
        # ).bind_tools([CodeReviewResult])
        llm = OllamaFunctions(model="exaone3.5:7.8b", format="json").bind_tools([CodeReviewResult])  # for local testing
        chain = prompt | llm

        try:
            chat_response = chain.invoke({"changes": changes})
            if chat_response.response_metadata:
                color.yellow(chat_response.response_metadata)

            code_review_result = CodeReviewResult.model_validate(chat_response.tool_calls[0]["args"])
            return code_review_result
        except Exception as e:
            color.red(e)
            return None
