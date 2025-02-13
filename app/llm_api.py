import re
from typing import Generator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.utils import clean_chat_response


class LlmAPI:

    @classmethod
    def review_code_changes_stream(
        cls,
        llm_model: str,
        changes: str,
    ) -> Generator[str, None, None]:
        system_message = """
You are a meticulous and highly skilled code reviewer. Your task is to analyze the following code changes and provide feedback **ONLY IF YOU IDENTIFY ANY OF THE FOLLOWING CRITICAL ISSUES**:
- Inappropriate function or variable names that are misleading, unclear, or violate naming conventions, making the code harder to understand or maintain.
- Security vulnerabilities, such as injection risks, improper authentication, data leaks, or insecure dependencies.
- Performance bottlenecks, including inefficient algorithms, redundant computations, memory leaks, or unnecessary resource usage.
        
Summarize your answers to make them as readable as possible. If you identify any issues, please also provide improvements or alternative solutions to address the problem, ensuring that your feedback is actionable and leads to a better solution.
Always respond in Korean. Do not use any other language unless explicitly asked.
""".strip()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", "{changes}"),
            ]
        )
        llm = ChatOllama(model=llm_model, temperature=0.5)
        chain = prompt | llm | StrOutputParser()
        for token in chain.stream({"changes": changes}):
            yield token

    @classmethod
    def check_any_issues(cls, llm_model: str, review_results: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Please read the following code review results. If no issues are found, MUST just respond with `LGTM`. If there are any issues, MUST just respond with `HAS_ISSUE`.",
                ),
                ("human", "{review_results}"),
            ]
        )
        llm = ChatOllama(model=llm_model, temperature=0)
        chain = prompt | llm | StrOutputParser()

        chat_response = chain.invoke({"review_results": clean_chat_response(review_results)})
        cleaned_chat_response = clean_chat_response(chat_response)

        if "HAS_ISSUE" in cleaned_chat_response:
            return True
        return False
