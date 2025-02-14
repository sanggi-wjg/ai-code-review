import io
import os
import re
from typing import Dict, List, Tuple

from unidiff import PatchSet, PatchedFile


def split_pr_diff_by_file(diff: str) -> Dict[str, str]:
    diff_by_file = {}
    files = re.split(r'(diff --git a/.*? b/.*?)\n', diff)[1:]

    for i in range(0, len(files), 2):
        file_header = files[i]
        file_diff = files[i + 1]

        match = re.search(r'diff --git a/(.*?) b/\1', file_header)
        if match:
            filename = match.group(1)
            diff_by_file[filename] = file_diff.strip()

    return diff_by_file


def split_patch_files_by_patch_types(diff: str) -> Tuple[List[PatchedFile], List[PatchedFile]]:
    target_extensions = (".py", ".kt", ".ts", ".tsx", ".js", ".jsx")
    ignore_filenames = ("Test", "test")

    def is_target_file(file: PatchedFile) -> bool:
        try:
            filename = os.path.basename(file.path)
            extension = os.path.splitext(file.path)[1]
        except IndexError:
            return False

        return all(
            [
                extension in target_extensions,
                not any(filename.__contains__(ignore) for ignore in ignore_filenames),
                file.added > 0,
            ]
        )

    patch_set = PatchSet(io.StringIO(diff))
    added_patch_files = [file for file in patch_set.added_files if is_target_file(file)]
    modified_patch_files = [file for file in patch_set.modified_files if is_target_file(file)]
    return added_patch_files, modified_patch_files


def clean_chat_response(chat_response: str):
    return re.sub(r'<think>.*?</think>', '', chat_response, flags=re.DOTALL).strip()
