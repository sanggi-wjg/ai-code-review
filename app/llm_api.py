import logging
import traceback
from typing import List, Optional, Generator, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeReviewIssue(BaseModel):
    """Represents a specific issue found during code review."""

    category: str = Field(
        description="Issue category",
        enum=["readability", "security", "performance", "best_practices"],
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
                "passed": "# ✅ 코드 리뷰 완료 😎",
                "needs_changes": "# ⚠️ 수정이 필요한 사항이 있습니다 ⚠️",
                "critical_issues": "# 🚨 중요한 문제가 발견 되었습니다 🚨",
            }.get(status, "# 🤖 코드 리뷰 완료 🤖")

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
You are a world-class software engineer specializing in **code quality, security, and performance optimization**. Your task is to **thoroughly review the given code** and provide **clear, actionable feedback**.

<IMPORTANT>
**Always respond in Korean.** Do not use any other language unless explicitly requested.
</IMPORTANT>

<Review Guideline>
Your feedback must be:
- **Precise & Concise**: Avoid vague comments. Always explain *why* an issue matters.
- **Constructive & Actionable**: Suggest improvements with clear justifications.
- **Structured & Readable**: Summarize findings clearly.
- **If the code has no issues**, explicitly state that it meets high standards and explain why.  
- **If issues exist**, provide detailed explanations and concrete solutions.

## **Review Criteria**
Analyze the code based on the following factors:

### **Readability & Maintainability**
- Are function/variable names clear and self-explanatory?
- Is the code modular and easy to understand?
- Are comments and documentation appropriate?

### **Security & Vulnerabilities**
- Are there any risks of **SQL Injection, XSS, CSRF, Hardcoded Secrets, or Authentication flaws**?
- Is sensitive data properly handled and encrypted?

### **Performance & Optimization**
- Are there **redundant computations** or **inefficient algorithms**?
- Are there **memory leaks or excessive resource consumption**?
- Could the logic be optimized for speed or scalability?

### **Best Practices & Code Consistency**
- Does the code follow standard conventions for the given programming language?
- Are there **anti-patterns, excessive nesting, or complex logic that can be simplified**?
</Review Guideline>

<OUTPUT_FORMAT>
Return the response in the following structured JSON format.

✅ If the Code is Well-Written:
{{
  "summary": "The code follows best practices, is well-structured, and has no security or performance issues.",
  "issues": [],
  "has_issues": false,
  "review_status": "passed"
}}

⚠️ If Issues Exist:
{{
  "summary": "The code is functional but has security vulnerabilities and inefficient loops.",
  "issues": [
    {{
      "category": "security",
      "description": "User input is directly concatenated into an SQL query, leading to SQL injection risks.",
      "suggestion": "Use parameterized queries or prepared statements to prevent SQL injection.",
      "severity": "high"
    }},
    {{
      "category": "performance",
      "description": "A nested loop results in O(n²) complexity, which is inefficient for large inputs.",
      "suggestion": "Refactor the algorithm using a hash map to reduce complexity to O(n).",
      "severity": "medium"
    }}
  ],
  "has_issues": true,
  "review_status": "needs_changes"
}}
</OUTPUT_FORMAT>
""".strip()

    SYSTEM_MESSAGE_CODING_ASSIST = """
You are a highly skilled software engineer specializing in developing secure and high-performance backend systems. Your goal is to generate optimized, well-structured, and maintainable code.
Always respond in Korean. Do not use any other language unless explicitly asked.""".strip()

    @classmethod
    def chat_to_review_code(cls, changes: str) -> Optional[CodeReviewResult]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", cls.SYSTEM_MESSAGE_CODE_REVIEW),
                ("human", "{changes}"),
            ]
        )
        llm = ChatOllama(
            model="qwen2.5:14b-instruct-q8_0",
            temperature=0.3,
            top_k=20,
            top_p=0.8,
        ).with_structured_output(CodeReviewResult)
        chain = prompt | llm
        try:
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
