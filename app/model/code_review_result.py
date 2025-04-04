from typing import List

from pydantic import BaseModel, Field


class CodeReviewIssue(BaseModel):
    """Represents a specific issue found during code review, aligned with structured review guidelines."""

    category: str = Field(
        description="Category of the issue, aligned with the review guideline step",
        enum=["code_quality", "functionality_correctness", "performance", "security_compliance"],
    )
    description: str = Field(description="Detailed explanation of the issue, including why it matters")
    suggestion: str = Field(
        description="Concrete suggestion to resolve the issue, including code examples where possible"
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

        def get_status_header(status: str) -> str:
            return {
                "passed": "# ✅ 코드 리뷰 완료 😎",
                "needs_changes": "# ⚠️ 수정이 필요한 사항이 있습니다 ⚠️",
                "critical_issues": "# 🚨 중요한 문제가 발견 되었습니다 🚨",
            }.get(status, "# 🤖 코드 리뷰 완료 🤖")

        def get_issue_category_title(category_type: str) -> str:
            return {
                "code_quality": "## 🧹 코드 품질 이슈",
                "functionality_correctness": "## 🔍 정확성 이슈",
                "performance": "## 🐢 성능 이슈",
                "security_compliance": "## 🔒 보안 이슈",
            }.get(category_type, "## 🛠️ 이슈")

        def get_severity_emoji(severity: str) -> str:
            return {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(severity, "❓")

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
