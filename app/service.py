import os.path
from typing import Generator, List

from colorful_print import color
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
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
    def chat_to_coding_assist(
        cls,
        code: str,
        repository: str,
        language: str,
        search: str,
        consideration: str,
    ) -> Generator[str, None, None]:
        vector_db = cls.get_vector_store(repository, language)
        documents = vector_db.similarity_search(query=search, k=5)
        return LlmAPI.chat_to_coding_assist_stream(
            documents,
            code,
            consideration,
        )

    @classmethod
    def chat_to_generate_code(
        cls,
        code: str,
        repository: str,
        language: str,
        search: str,
        consideration: str,
    ) -> Generator[str, None, None]:
        vector_db = cls.get_vector_store(repository, language)
        documents = vector_db.similarity_search(query=search, k=5)
        return LlmAPI.chat_to_generate_code_stream(
            documents,
            code,
            consideration,
        )

    @classmethod
    def _load_documents(cls, repository: str, language: str) -> List[Document]:
        source_path = os.path.join(os.getcwd(), "sources", repository)
        GithubAPI.clone_or_pull(repository, source_path)

        if language == Language.PYTHON:
            loader = DirectoryLoader(path=source_path, glob="**/*.py")
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=512,
                chunk_overlap=50,
            )
        elif language == Language.KOTLIN:
            loader = DirectoryLoader(path=source_path, glob="**/*.kt")
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.KOTLIN,
                chunk_size=512,
                chunk_overlap=50,
            )
        else:
            raise ValueError(f"Unsupported language: {language}")

        documents = loader.load_and_split(splitter)
        return documents

    @classmethod
    def _get_embeddings(cls, repository: str) -> Embeddings:
        embeddings = OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=LocalFileStore(f"embeddings.cache.{repository}"),
            namespace=f"{embeddings.model}.{repository}",
        )
        return cached_embeddings

    @classmethod
    def get_vector_store(cls, repository: str, language: str) -> VectorStore:
        repository_replaced = repository.replace("/", ".")

        documents = cls._load_documents(repository, language)
        embeddings = cls._get_embeddings(repository_replaced)
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=f"chroma.{repository_replaced}",
        )

    @classmethod
    def index(cls, repository: str, language: str):
        return cls.get_vector_store(repository, language)
