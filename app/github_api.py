import os

import git
import requests
from colorful_print import color


class GithubAPI:

    @classmethod
    def clone_or_pull(cls, repository: str, source_path: str) -> bool:
        url = f"https://github.com/{repository}.git"

        if os.path.exists(source_path):
            git.Repo(source_path).git.pull()
        else:
            git.Repo.clone_from(url, source_path)
        return True

    @classmethod
    def get_pr(
        cls,
        token: str,
        repository: str,
        pr_number: int,
    ):
        try:
            response = requests.get(
                url=f"https://api.github.com/repos/{repository}/pulls/{pr_number}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Accept": "application/vnd.github+json",
                },
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            raise e

    @classmethod
    def get_pr_diff(
        cls,
        token: str,
        repository: str,
        pr_number: int,
    ) -> requests.Response:
        try:
            response = requests.get(
                url=f"https://{token}:x-oauth-basic@api.github.com/repos/{repository}/pulls/{pr_number}",
                headers={"Accept": "application/vnd.github.v3.diff"},
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            raise e

    @classmethod
    def show_rate_limit(cls, token: str):
        try:
            response = requests.get(
                url="https://api.github.com/rate_limit",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            response.raise_for_status()
            color.cyan(response.json(), italic=True)
        except requests.HTTPError as e:
            raise e

    @classmethod
    def create_review_comment(
        cls,
        token: str,
        repository: str,
        pr_number: int,
        comment: str,
        commit_id: str,
        filename: str,
        start_line: int,
        end_line: int,
        side: str,
    ) -> requests.Response:
        """
        https://docs.github.com/en/rest/reference/pulls#create-a-review-comment
        """
        try:
            response = requests.post(
                url=f"https://api.github.com/repos/{repository}/pulls/{pr_number}/comments",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "Accept": "application/vnd.github+json",
                },
                json={
                    "body": comment,
                    "commit_id": commit_id,
                    "path": filename,
                    "start_line": start_line,
                    "line": end_line,
                    "side": side,
                },
            )
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            raise e
