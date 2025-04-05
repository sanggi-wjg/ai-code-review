import logging
import traceback
from typing import List, Optional, Generator, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.model.code_review_result import CodeReviewResult
from app.prompt import CODING_ASSIST_INSTRUCT, CODE_REVIEW_INSTRUCT_2

logger = logging.getLogger(__name__)


class LlmAPI:

    @classmethod
    def chat_to_review_code(cls, changes: str) -> Optional[CodeReviewResult]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CODE_REVIEW_INSTRUCT_2),
                ("human", "{changes}"),
            ]
        )
        llm = ChatOllama(
            model="qwen2.5:32b",
            # model="gemma3:27b",
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
{consideration}
""".strip()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CODING_ASSIST_INSTRUCT),
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
