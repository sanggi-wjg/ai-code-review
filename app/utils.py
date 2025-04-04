import abc
import functools
import io
import os
import pickle
import re
from typing import Dict, List, Tuple, Any, Type, Callable

import redis
from unidiff import PatchSet, PatchedFile


def split_pr_diff_by_file(diff: str) -> Dict[str, str]:
    diff_by_file = {}

    parts = re.split(r'(?=^diff --git a/.*? b/.*?)', diff, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue

        match = re.search(r'diff --git a/(.*?) b/(.*?)\n', part)
        if match:
            filename = match.group(2)
            diff_by_file[filename] = part.strip()

    # files = re.split(r'(diff --git a/.*? b/.*?)\n', diff)[1:]
    #
    # for i in range(0, len(files), 2):
    #     file_header = files[i]
    #     file_diff = files[i + 1]
    #
    #     match = re.search(r'diff --git a/(.*?) b/\1', file_header)
    #     if match:
    #         filename = match.group(1)
    #         diff_by_file[filename] = file_diff.strip()

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


class CacheHandler(abc.ABC):

    def __init__(self, key: str):
        self.key = key

    @abc.abstractmethod
    def is_cached(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def read(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def write(self, dataset: Any):
        raise NotImplementedError()


class RedisCacheHandler(CacheHandler):

    def __init__(self, key: str):
        super().__init__(key)
        self.client = redis.StrictRedis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=os.getenv("REDIS_PORT", 6379),
            db=os.getenv("REDIS_DB", 1),
        )
        self.expire_ttl = os.getenv("REDIS_EXPIRE_TTL", 60 * 60)

    def is_cached(self) -> bool:
        return self.client.exists(self.key)

    def read(self) -> Any:
        return pickle.loads(
            self.client.get(self.key),
        )

    def write(self, dataset: Any):
        self.client.set(
            name=self.key,
            value=pickle.dumps(dataset),
            ex=self.expire_ttl,
        )


def cacheable(key: str, cache_handler: Type[CacheHandler] = RedisCacheHandler):

    def decorator(func: Callable):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = cache_handler(key)
            if handler.is_cached():
                return handler.read()

            handler.write(result := func(*args, **kwargs))
            return result

        return wrapper

    return decorator
