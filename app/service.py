import time

from colorful_print import color
from unidiff import Hunk, PatchedFile

from app.github_api import GithubAPI
from app.llm_api import LlmAPI
from app.utils import split_pr_diff_by_file, split_patch_files_by_patch_types


class CodeReviewService:

    @classmethod
    def review(
        cls,
        github_token: str,
        groq_api_key: str,
        groq_model: str,
        repository: str,
        pr_number: int,
    ):
        pr_response = GithubAPI.get_pr(github_token, repository, pr_number)
        head_commit_id = pr_response.json()["head"]["sha"]

        pr_diff_response = GithubAPI.get_pr_diff(github_token, repository, pr_number)
        diff_by_file = split_pr_diff_by_file(pr_diff_response.text)

        added_patch_files, modified_patch_files = split_patch_files_by_patch_types(pr_diff_response.text)

        for patch in added_patch_files:
            hunk: Hunk
            hunk = patch[0]
            start_line, end_line = hunk.target_start, hunk.target_length
            color.green(f"\n{patch.path}, review start", bold=True, itailic=True, underline=True)

            cls._review_and_left_comment(
                github_token=github_token,
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                repository=repository,
                pr_number=pr_number,
                head_commit_id=head_commit_id,
                diff=diff_by_file[patch.path],
                patch=patch,
                start_line=start_line,
                end_line=end_line,
            )

        for patch in modified_patch_files:
            hunk: Hunk
            hunk = max(patch, key=lambda x: x.added)
            start_line, end_line = hunk.target_start, hunk.target_start + hunk.added
            color.green(f"\n{patch.path}, review start", bold=True, itailic=True, underline=True)

            cls._review_and_left_comment(
                github_token=github_token,
                groq_api_key=groq_api_key,
                groq_model=groq_model,
                repository=repository,
                pr_number=pr_number,
                head_commit_id=head_commit_id,
                diff=diff_by_file[patch.path],
                patch=patch,
                start_line=start_line,
                end_line=end_line,
            )

    @classmethod
    def _review_and_left_comment(
        cls,
        github_token: str,
        groq_api_key: str,
        groq_model: str,
        repository: str,
        pr_number: int,
        head_commit_id: str,
        diff: str,
        patch: PatchedFile,
        start_line: int,
        end_line: int,
    ):
        review_result = LlmAPI.chat_to_review_code(groq_api_key, groq_model, diff)
        if review_result is None or not review_result.has_issues:
            return

        GithubAPI.create_review_comment(
            token=github_token,
            repository=repository,
            pr_number=pr_number,
            comment=review_result.format_to_comment(),
            commit_id=head_commit_id,
            filename=patch.path,
            start_line=start_line,
            end_line=end_line,
            side="RIGHT",
        )
        # time.sleep(60)


class CodeChatService:

    @classmethod
    def chat(cls, code: str) -> str:
        return LlmAPI.chat_to_coding_assist(code)
