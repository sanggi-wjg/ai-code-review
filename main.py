import io
import os

from colorful_print import color
from dotenv import load_dotenv
from unidiff import PatchSet, Hunk, PatchedFile

from app.github_api import GithubAPI
from app.llm_api import LlmAPI
from app.utils import split_pr_diff_by_file, clean_chat_response, split_patch_files_by_patch_types

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def review_and_comment(
    repository: str,
    pr_number: int,
    model: str,
    head_commit_id: str,
    diff: str,
    patch: PatchedFile,
    start_line: int,
    end_line: int,
):
    code_review = ""
    for token in LlmAPI.review_code_changes_stream(model, diff):
        color.green(token, end="", flush=True)
        code_review += token

    GithubAPI.create_review_comment(
        token=GITHUB_TOKEN,
        repository=repository,
        pr_number=pr_number,
        comment=clean_chat_response(code_review),
        commit_id=head_commit_id,
        filename=patch.path,
        start_line=start_line,
        end_line=end_line,
        side="RIGHT",
    )


def main(
    repository: str,
    pr_number: int,
    model: str,
):
    GithubAPI.show_rate_limit(GITHUB_TOKEN)

    pr_response = GithubAPI.get_pr(GITHUB_TOKEN, repository, pr_number).json()
    head_commit_id = pr_response["head"]["sha"]

    pr_diff_response = GithubAPI.get_pr_diff(GITHUB_TOKEN, repository, pr_number)
    diff_by_file = split_pr_diff_by_file(pr_diff_response.text)

    added_patch_files, modified_patch_files = split_patch_files_by_patch_types(pr_diff_response.text)

    for patch in added_patch_files:
        hunk: Hunk
        hunk = patch[0]
        color.yellow(f"\n{patch.path}, review start", bold=True, itailic=True, underline=True)

        review_and_comment(
            repository=repository,
            pr_number=pr_number,
            model=model,
            head_commit_id=head_commit_id,
            diff=diff_by_file[patch.path],
            patch=patch,
            start_line=hunk.target_start,
            end_line=hunk.target_length,
        )

    for patch in modified_patch_files:
        hunk: Hunk
        hunk = max(patch, key=lambda x: x.added)
        color.yellow(f"\n{patch.path}, review start", bold=True, itailic=True, underline=True)

        review_and_comment(
            repository=repository,
            pr_number=pr_number,
            model=model,
            head_commit_id=head_commit_id,
            diff=diff_by_file[patch.path],
            patch=patch,
            start_line=hunk.target_start,
            end_line=hunk.target_start + hunk.target_length - 1,
        )


if __name__ == '__main__':
    main("FitpetKorea/fitpetmall-backend-v4", 2652, "deepseek-r1:14b")
