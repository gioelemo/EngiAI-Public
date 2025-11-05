"""
RAG Agent for document-based question answering.

This agent handles queries about uploaded documents using the RAG system.
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from config import config
from src.checkpoint import get_checkpointer
from src.models.state import MessagesState
from src.tools import EngineeringRAGChain, EngineerRAGStore, MultimodalDocumentProcessor

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agent for document-based question answering using RAG."""

    def __init__(self, model_name: str | None = None):
        """Initialize the RAG agent with vector store and tools.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        self.model_name = model_name or config.llm_model
        self.llm = init_chat_model(self.model_name)

        # Initialize RAG components
        self.document_processor = MultimodalDocumentProcessor()
        self.vector_store = EngineerRAGStore(collection_name="engineer_docs")
        self.rag_chain = EngineeringRAGChain(self.vector_store)

        # Set up tools
        self.tools = self._create_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the agent graph
        self.agent = self._build_agent()
        logger.info("RAG Agent initialized with vector store")

    def _create_tools(self) -> list:
        """Create LangChain tools for the RAG agent."""
        return [
            self._create_search_tool(),
            self._create_add_document_tool(),
            self._create_list_documents_tool(),
            self._create_clear_memory_tool(),
        ]

    def _create_search_tool(self):
        """Create the search documents tool."""

        @tool
        def search_documents(
            query: Annotated[str, "The question to search for in documents"],
            num_results: Annotated[int, "Number of relevant documents to retrieve"] = 5,
        ) -> str:
            """
            Search through uploaded documents to find relevant information.

            Use this tool to answer questions about papers, technical documents,
            or any previously uploaded files. It returns contextual information
            with source citations.
            """
            try:
                result = self.rag_chain.ask(query, k=num_results)
                answer = result["answer"]
                sources = result["num_sources"]

            except Exception as e:
                logger.exception("Error searching documents")
                return f"Error searching documents: {e}"

            else:
                # Format response with source information
                response = f"{answer}\n\n📚 Sources: {sources} document(s)"
                return response

        return search_documents

    def _create_add_document_tool(self):
        """Create the add document tool."""

        @tool
        def add_document(
            file_path: Annotated[str, "Path to the document file to add"],
            metadata: Annotated[str, "Optional metadata as JSON string"] = "{}",
        ) -> str:
            """
            Add a new document to the knowledge base.

            Processes PDF files and adds them to the vector store for future queries.
            The document will be chunked and embedded automatically.
            """
            try:
                meta = json.loads(metadata) if metadata != "{}" else {}

                # Process document
                docs = self.document_processor.process_file(file_path)

                # Add metadata
                for doc in docs:
                    doc.metadata.update(meta)

                # Store in vector database
                self.vector_store.add_documents(docs)

                file_name = Path(file_path).name
                return f"✓ Successfully added '{file_name}' to knowledge base ({len(docs)} chunks)"

            except Exception as e:
                logger.exception("Error adding document")
                return f"Error adding document: {e}"

        return add_document

    def _create_list_documents_tool(self):
        """Create the list documents tool."""

        @tool
        def list_documents() -> str:
            """
            List all documents currently in the knowledge base.

            Returns a summary of stored documents with their sources and page counts.
            """
            try:
                # Get all documents (limited retrieval to avoid overload)
                all_docs = self.vector_store.similarity_search("", k=100)

                if not all_docs:
                    return "No documents in the knowledge base yet."

                # Organize by source
                sources: dict[str, dict[str, Any]] = {}
                for doc in all_docs:
                    source = doc.metadata.get("source", "unknown")
                    if source not in sources:
                        sources[source] = {"pages": set(), "chunks": 0}

                    sources[source]["chunks"] += 1
                    if "page" in doc.metadata:
                        sources[source]["pages"].add(doc.metadata["page"])

                # Format output
                result = f"📚 Knowledge Base ({len(sources)} documents):\n\n"
                for source, info in sources.items():
                    file_name = Path(source).name
                    pages_set: set[Any] = info["pages"]  # type: ignore[assignment]
                    pages = len(pages_set) if pages_set else "N/A"
                    chunks: int = info["chunks"]  # type: ignore[assignment]
                    result += f"• {file_name}\n  - Pages: {pages}, Chunks: {chunks}\n"

                total_count = self.vector_store.get_collection_count()
                result += f"\nTotal chunks: {total_count}"

            except Exception as e:
                logger.exception("Error listing documents")
                return f"Error listing documents: {e}"

            else:
                return result

        return list_documents

    def _create_clear_memory_tool(self):
        """Create the clear memory tool."""

        @tool
        def clear_document_memory() -> str:
            """
            Clear the conversation history for document Q&A.

            Use this when starting a new topic or when the user wants to reset
            the conversation context.
            """
            try:
                self.rag_chain.clear_history()
            except Exception as e:
                logger.exception("Error clearing history")
                return f"Error clearing history: {e}"
            else:
                return "✓ Conversation history cleared"

        return clear_document_memory

    def _llm_call(self, state: MessagesState) -> dict:
        """LLM decides whether to call a tool or not.

        Args:
            state: Current conversation state

        Returns:
            Updated state with LLM response
        """
        system_prompt = """You are a specialized document assistant for engineering research.

Your role is to help users understand and extract information from technical documents,
research papers, and engineering specifications they have uploaded.

Guidelines:
1. **Always cite sources**: Include document names and page numbers when answering
2. **Be precise**: Engineering work requires accuracy - cite specific sections
3. **Ask for clarification**: If a question is ambiguous, ask for more details
4. **Acknowledge limitations**: If information isn't in the documents, say so clearly
5. **Use conversation history**: Reference previous questions for better context
6. **Suggest related topics**: When appropriate, suggest related questions users might ask

When users upload documents:
- Confirm successful processing
- Provide a brief summary of what was added
- Suggest 2-3 initial questions they could ask about the document

Always be helpful, accurate, and cite your sources!"""

        messages: list[AnyMessage] = [SystemMessage(content=system_prompt)] + state[
            "messages"
        ]

        return {
            "messages": [self.llm_with_tools.invoke(messages)],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def _tool_node(self, state: MessagesState) -> dict:
        """Execute tool calls from the LLM.

        Args:
            state: Current conversation state

        Returns:
            Updated state with tool results
        """
        result = []
        last_message = state["messages"][-1]

        # Only AIMessage has tool_calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                result.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
                )

        return {"messages": result}

    def _should_continue(self, state: MessagesState) -> Literal["tool_node", "__end__"]:
        """Decide whether to continue to tool execution or end.

        Args:
            state: Current conversation state

        Returns:
            Next node to execute ("tool_node" or "__end__")
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return "__end__"

    def _build_agent(self) -> Any:
        """Build the agent workflow graph.

        Returns:
            Compiled agent graph
        """
        # Build workflow
        agent_builder = StateGraph(MessagesState)

        # Add nodes
        agent_builder.add_node("llm_call", self._llm_call)
        agent_builder.add_node("tool_node", self._tool_node)

        # Add edges to connect nodes
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call", self._should_continue, ["tool_node", END]
        )
        agent_builder.add_edge("tool_node", "llm_call")

        # Compile with persistent checkpointer for conversation memory
        checkpointer = get_checkpointer()
        return agent_builder.compile(checkpointer=checkpointer)

    def invoke(self, state: MessagesState, config: dict | None = None) -> MessagesState:
        """Invoke the agent with a given state.

        Args:
            state: Input conversation state
            config: Optional configuration (e.g., thread_id for conversation tracking)

        Returns:
            Updated conversation state
        """
        return self.agent.invoke(state, config)  # type: ignore[return-value]
