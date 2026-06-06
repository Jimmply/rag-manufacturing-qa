"""
LangChain Q&A chain for the Manufacturing Document RAG system.

Supports Anthropic (claude-sonnet-4-6) and OpenAI (gpt-4o) as backends,
selected via the MODEL_PROVIDER environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Generator, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from retriever import ManufacturingRetriever

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are a knowledgeable manufacturing domain expert specializing in industrial \
equipment maintenance, laser cutting systems, and operational safety. Your role is to answer \
questions based strictly on the provided documentation excerpts.

Guidelines:
- Answer only based on the retrieved context provided. Do not fabricate specifications, \
part numbers, error codes, or procedures.
- If the context does not contain enough information to answer the question, say so clearly \
and suggest what type of document might contain the answer.
- When citing procedures or specifications, reference the source document and section \
where applicable.
- Use clear, technical language appropriate for maintenance engineers and operators.
- For safety-critical questions, always emphasize relevant warnings and PPE requirements \
from the documentation.
- Structure longer answers with numbered steps or bullet points for clarity.

Context from retrieved documents:
{context}"""


def _build_llm() -> BaseChatModel:
    """Instantiate the configured LLM backend."""
    if MODEL_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            anthropic_api_key=api_key,
            max_tokens=2048,
            temperature=0.1,
        )
    elif MODEL_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=api_key,
            max_tokens=2048,
            temperature=0.1,
        )
    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER: '{MODEL_PROVIDER}'. Must be 'anthropic' or 'openai'."
        )


class ManufacturingQAChain:
    """
    RAG Q&A chain that retrieves relevant document chunks and generates
    grounded answers with source citations.
    """

    def __init__(self, retriever: ManufacturingRetriever | None = None) -> None:
        self.retriever = retriever or ManufacturingRetriever()
        self._llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = _build_llm()
        return self._llm

    def answer(
        self,
        question: str,
        chat_history: List[BaseMessage] | None = None,
    ) -> dict:
        """
        Generate an answer for a question using RAG.

        Returns a dict with:
          - answer (str): The LLM response
          - sources (List[dict]): Structured source citation metadata
          - context (str): The raw context passed to the LLM
        """
        results = self.retriever.retrieve(question)
        context = self.retriever.format_context(results)
        sources = self.retriever.get_source_citations(results)

        messages: List[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        ]

        if chat_history:
            messages.extend(chat_history[-6:])

        messages.append(HumanMessage(content=question))

        logger.debug("Sending %d messages to %s/%s", len(messages), MODEL_PROVIDER, ANTHROPIC_MODEL if MODEL_PROVIDER == "anthropic" else OPENAI_MODEL)

        response: AIMessage = self.llm.invoke(messages)

        retrieval_confidence = self._compute_confidence(results)

        return {
            "answer": response.content,
            "sources": sources,
            "context": context,
            "retrieval_confidence": retrieval_confidence,
        }

    def stream_answer(
        self,
        question: str,
        chat_history: List[BaseMessage] | None = None,
    ) -> Generator[str, None, dict]:
        """
        Stream the LLM response token-by-token.

        Yields string tokens. The final item yielded is a dict with source metadata
        (check isinstance(item, dict) to detect it).
        """
        results = self.retriever.retrieve(question)
        context = self.retriever.format_context(results)
        sources = self.retriever.get_source_citations(results)

        messages: List[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        ]

        if chat_history:
            messages.extend(chat_history[-6:])

        messages.append(HumanMessage(content=question))

        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

        yield {
            "sources": sources,
            "context": context,
            "retrieval_confidence": self._compute_confidence(results),
        }

    @staticmethod
    def _compute_confidence(results: list) -> float:
        """
        Estimate retrieval confidence from similarity scores.

        Returns a 0–1 float: mean similarity of the top retrieved chunks,
        clipped so values below 0.3 are treated as low-confidence.
        """
        if not results:
            return 0.0
        scores = []
        for doc in results:
            score = None
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                score = doc.metadata.get("score") or doc.metadata.get("relevance_score")
            if score is None and isinstance(doc, tuple) and len(doc) == 2:
                score = doc[1]
            if score is not None:
                scores.append(float(score))
        if not scores:
            return 0.5  # unknown — assume moderate confidence
        return round(min(1.0, max(0.0, float(sum(scores) / len(scores)))), 3)

    def change_provider(self, provider: str, model: str | None = None) -> None:
        """Hot-swap the LLM provider without rebuilding the retriever."""
        global MODEL_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL
        MODEL_PROVIDER = provider.lower()
        if model:
            if provider == "anthropic":
                ANTHROPIC_MODEL = model
            else:
                OPENAI_MODEL = model
        self._llm = None
        logger.info("Switched to provider=%s, model=%s", MODEL_PROVIDER, model)
