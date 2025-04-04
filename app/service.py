import logging
import os.path
from typing import Generator, List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from unidiff import Hunk, PatchedFile

from app.github_api import GithubAPI
from app.llm_api import LlmAPI
from app.utils import split_pr_diff_by_file, split_patch_files_by_patch_types

logger = logging.getLogger(__name__)


class CodeReviewService:

    @classmethod
    def review(
        cls,
        github_token: str,
        repository: str,
        pr_number: int,
    ):
        logger.info(f"🚀🚀 Repository: {repository} / Pull Request: {pr_number} / start reviewing  🚀🚀")

        pr_response = GithubAPI.get_pr(github_token, repository, pr_number)
        head_commit_id = pr_response.json()["head"]["sha"]

        pr_diff_response = GithubAPI.get_pr_diff(github_token, repository, pr_number)
        diff_by_file = split_pr_diff_by_file(pr_diff_response.text)

        added_patch_files, modified_patch_files = split_patch_files_by_patch_types(pr_diff_response.text)

        for patch in added_patch_files:
            hunk: Hunk = patch[0]
            start_line, end_line = hunk.target_start, hunk.target_length
            logger.info(f"🤖🤖 {patch.path}, review start")

            cls._review_and_left_comment(
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
            hunk: Hunk = max(patch, key=lambda x: x.added)
            start_line, end_line = hunk.target_start, hunk.target_start + hunk.target_length
            logger.info(f"🤖🤖 {patch.path}, review start")

            cls._review_and_left_comment(
                github_token=github_token,
                repository=repository,
                pr_number=pr_number,
                head_commit_id=head_commit_id,
                diff=diff_by_file[patch.path],
                patch=patch,
                start_line=start_line,
                end_line=end_line,
            )

        logger.info(f"😎👍 {repository}:{pr_number} review end 😎👍")

    @classmethod
    def _review_and_left_comment(
        cls,
        github_token: str,
        repository: str,
        pr_number: int,
        head_commit_id: str,
        diff: str,
        patch: PatchedFile,
        start_line: int,
        end_line: int,
    ):
        review_result = LlmAPI.chat_to_review_code(diff)
        logger.info(f"Review result: {review_result}")
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


class CodeChatService:

    @classmethod
    def chat_about_repository(
        cls,
        repository: str,
        search: str,
    ) -> dict:
        vector_store = cls.get_vector_store(repository)
        found_documents = vector_store.similarity_search_with_relevance_scores(
            search,
            k=5,
            score_threshold=0.75,
        )
        logger.info(f"{found_documents}")
        return {
            "documents": found_documents,
            "answer": LlmAPI.chat_to_ask(found_documents),
        }

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

        ignore_filenames = ("Test", "test")
        documents = [
            doc
            for doc in loader.load()
            if not any(ignore_filename in doc.metadata["source"] for ignore_filename in ignore_filenames)
        ]
        return splitter.split_documents(documents)

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
            index_params=[
                {
                    "index_name": "index_vector",
                    "field_name": "vector",
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                }
            ],
            enable_dynamic_field=True,
            auto_id=True,
            drop_old=drop_old,
        )

    @classmethod
    def index(cls, repository: str, language: str) -> List[str]:
        logger.info("Index start")

        documents = cls.load_documents_from(repository, language)
        vector_store = cls.get_vector_store(repository, drop_old=True)
        created_ids = vector_store.add_documents(documents)

        logger.info("Index finished")
        return created_ids
