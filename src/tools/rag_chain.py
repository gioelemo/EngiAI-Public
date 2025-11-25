"""
RAG chain for question-answering with conversation memory.

Combines retrieval, prompt engineering, and LLM generation for engineering Q&A.
"""

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from config import config
from src.tools.vector_store import EngineerRAGStore

logger = logging.getLogger(__name__)


if STREAMLIT_AVAILABLE:

    @st.cache_resource
    def get_shared_rag_chain(
        _vector_store: EngineerRAGStore,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> "EngineeringRAGChain":
        """Get or create a shared RAG chain instance (cached across all users/sessions).

        This is cached with @st.cache_resource to avoid expensive re-initialization.
        The RAG chain is stateless (conversation history is managed per-agent) and
        can be safely shared globally.

        Note: _vector_store has underscore prefix to exclude from cache key hashing,
        since it's already a cached resource.

        Args:
            _vector_store: Shared vector store instance (excluded from cache key)
            model_name: LLM model name
            system_prompt: Custom system prompt
            temperature: LLM temperature

        Returns:
            Shared EngineeringRAGChain instance
        """
        logger.info("Creating/retrieving shared RAG chain")
        return EngineeringRAGChain(
            vectorstore=_vector_store,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
        )


class EngineeringRAGChain:
    """RAG chain for engineering document Q&A with context-aware responses."""

    # System prompt for engineering context
    DEFAULT_SYSTEM_PROMPT = """You are an expert engineering assistant with access to technical documentation.

Your role is to answer questions based on the provided context from engineering documents,
research papers, and technical specifications.

Guidelines:
- Answer questions based ONLY on the provided context
- If the context doesn't contain enough information, clearly state that
- Focus on technical accuracy and precision
- Cite specific sources when possible (mention document names, page numbers)
- For equations or formulas, preserve mathematical notation
- If multiple documents provide relevant information, synthesize them coherently

Context from documents:
{context}

Previous conversation:
{chat_history}"""

    def __init__(
        self,
        vectorstore: EngineerRAGStore,
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ):
        """Initialize the RAG chain.

        Args:
            vectorstore: Vector store for document retrieval
            model_name: LLM model name (defaults to config.llm_model)
            system_prompt: Custom system prompt (defaults to DEFAULT_SYSTEM_PROMPT)
            temperature: LLM temperature for generation (0 = deterministic)
        """
        self.vectorstore = vectorstore
        self.model_name = model_name or config.llm_model

        # Initialize LLM
        self.llm = init_chat_model(self.model_name, temperature=temperature)

        # Set up prompt template
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        # Conversation history (simple list for now)
        self.chat_history: list[dict] = []

        logger.info(f"RAG chain initialized with model: {self.model_name}")

    def _format_docs(self, docs: list[Document]) -> str:
        """Format documents for context in the prompt.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted string with document contents and metadata
        """
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            formatted.append(
                f"[Document {i}] (Source: {source}, Page: {page}, Chunk: {chunk_id})\n"
                f"{doc.page_content}\n"
            )

        return "\n".join(formatted)

    def _format_chat_history(self) -> str:
        """Format conversation history for context.

        Returns:
            Formatted chat history string
        """
        if not self.chat_history:
            return "No previous conversation."

        formatted = []
        for entry in self.chat_history[-5:]:  # Keep last 5 exchanges
            formatted.append(f"User: {entry['question']}")
            formatted.append(f"Assistant: {entry['answer']}\n")

        return "\n".join(formatted)

    def ask(
        self,
        question: str,
        k: int = 5,
        filter_dict: dict | None = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """Ask a question with RAG.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter for retrieval
            include_sources: Whether to include source documents in response

        Returns:
            Dictionary with answer, sources, and context
        """
        logger.info(f"Processing question: {question[:100]}...")

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(
            question, k=k, filter_dict=filter_dict
        )

        logger.debug(f"Retrieved {len(docs)} documents")

        # Format context and chat history
        context = self._format_docs(docs)
        chat_history = self._format_chat_history()

        # Build the chain
        chain: Any = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Get answer
        answer = chain.invoke(question)

        # Save to history
        self.chat_history.append({"question": question, "answer": answer})

        # Prepare response
        response: dict[str, Any] = {
            "answer": answer,
            "question": question,
        }

        if include_sources:
            response["source_documents"] = docs
            response["context"] = context
            response["num_sources"] = len(docs)

        logger.info("Question answered successfully")

        return response

    def ask_with_scores(
        self, question: str, k: int = 5, filter_dict: dict | None = None
    ) -> dict:
        """Ask a question and include relevance scores.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter for retrieval

        Returns:
            Dictionary with answer, sources with scores
        """
        logger.info(f"Processing question with scores: {question[:100]}...")

        # Retrieve with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=k, filter_dict=filter_dict
        )

        # Extract scores
        scores = [score for _, score in docs_with_scores]

        # Get answer using regular flow
        result = self.ask(question, k=k, filter_dict=filter_dict, include_sources=True)

        # Add scores
        result["scores"] = scores
        result["docs_with_scores"] = docs_with_scores

        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict]:
        """Get conversation history.

        Returns:
            List of question-answer pairs
        """
        return self.chat_history.copy()

    def stream_answer(self, question: str, k: int = 5, filter_dict: dict | None = None):
        """Stream the answer token by token (for real-time display).

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter for retrieval

        Yields:
            Answer tokens as they are generated
        """
        # Retrieve documents
        docs = self.vectorstore.similarity_search(
            question, k=k, filter_dict=filter_dict
        )

        context = self._format_docs(docs)
        chat_history = self._format_chat_history()

        # Build streaming chain
        chain: Any = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Stream response
        full_answer = ""
        for chunk in chain.stream(question):
            full_answer += chunk
            yield chunk

        # Save to history after streaming completes
        self.chat_history.append({"question": question, "answer": full_answer})
