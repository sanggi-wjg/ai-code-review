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
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from unidiff import Hunk, PatchedFile

from app.github_api import GithubAPI
from app.llm_api import LlmAPI, CodeReviewResult
from app.utils import split_pr_diff_by_file, split_patch_files_by_patch_types


class CodeReviewService:

    @classmethod
    def review(
        cls,
        model: str,
        github_token: str,
        repository: str,
        pr_number: int,
    ):
        color.green(f"{repository}:{pr_number} review start", underline=True)

        pr_response = GithubAPI.get_pr(github_token, repository, pr_number)
        head_commit_id = pr_response.json()["head"]["sha"]

        pr_diff_response = GithubAPI.get_pr_diff(github_token, repository, pr_number)
        diff_by_file = split_pr_diff_by_file(pr_diff_response.text)

        added_patch_files, modified_patch_files = split_patch_files_by_patch_types(pr_diff_response.text)

        for patch in added_patch_files:
            hunk: Hunk
            hunk = patch[0]
            start_line, end_line = hunk.target_start, hunk.target_length
            color.green(f"\n{patch.path}, review start", bold=True, underline=True)

            cls._review_and_left_comment(
                model=model,
                github_token=github_token,
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
            color.green(f"\n{patch.path}, review start", bold=True, underline=True)

            cls._review_and_left_comment(
                model=model,
                github_token=github_token,
                repository=repository,
                pr_number=pr_number,
                head_commit_id=head_commit_id,
                diff=diff_by_file[patch.path],
                patch=patch,
                start_line=start_line,
                end_line=end_line,
            )

        color.green(f"\n{repository}:{pr_number} review end", underline=True)

    @classmethod
    def _review_and_left_comment(
        cls,
        model: str,
        github_token: str,
        repository: str,
        pr_number: int,
        head_commit_id: str,
        diff: str,
        patch: PatchedFile,
        start_line: int,
        end_line: int,
    ):
        review_result = LlmAPI.chat_to_review_code(model, diff)
        if review_result is None:
            return

        color.yellow(review_result)
        if isinstance(review_result, CodeReviewResult):
            if not review_result.has_issues:
                return
            comment = review_result.format_to_comment()
        else:
            comment = review_result

        GithubAPI.create_review_comment(
            token=github_token,
            repository=repository,
            pr_number=pr_number,
            comment=comment,
            commit_id=head_commit_id,
            filename=patch.path,
            start_line=start_line,
            end_line=end_line,
            side="RIGHT",
        )


class CodeChatService:

    @classmethod
    def chat_about_repository(
        cls,
        repository: str,
        search: str,
    ) -> dict:
        vector_store = cls.get_vector_store(repository)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return LlmAPI.chat_to_ask(retriever, search)

    @classmethod
    def chat_to_generate_code(
        cls,
        code: str,
        repository: str,
        language: str,
        search: str,
        consideration: str,
    ) -> Generator[str, None, None]:
        vector_db = cls.get_vector_store(repository)
        documents = vector_db.similarity_search(query=search, k=5)
        return LlmAPI.chat_to_generate_code_stream(
            documents,
            code,
            consideration,
        )

    @classmethod
    def load_documents_from(cls, repository: str, language: str) -> List[Document]:
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
    def get_embeddings(cls, repository: str) -> Embeddings:
        repository_replaced = repository.replace("/", "_")

        embeddings = OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=LocalFileStore(f"embeddings.cache.{repository_replaced}"),
            namespace=f"{embeddings.model}.{repository_replaced}",
        )
        return cached_embeddings

    @classmethod
    def get_vector_store(cls, repository: str, drop_old: bool = False) -> VectorStore:
        def escape_collection_name(value: str) -> str:
            # If u need, using regex
            return value.replace("/", "_").replace("-", "_")

        repository_replaced = escape_collection_name(repository)
        return Milvus(
            embedding_function=cls.get_embeddings(repository),
            connection_args={"uri": os.getenv("MILVUS_URI", "http://localhost:19530")},
            collection_name=repository_replaced,
            collection_description=f"{repository} Vector Store",
            # metadata_field="metadata",
            enable_dynamic_field=True,
            auto_id=True,
            drop_old=drop_old,
        )

    @classmethod
    def index(cls, repository: str, language: str) -> List[str]:
        documents = cls.load_documents_from(repository, language)
        vector_store = cls.get_vector_store(repository, drop_old=True)
        return vector_store.add_documents(documents)
