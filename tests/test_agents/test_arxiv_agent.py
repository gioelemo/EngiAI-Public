"""Tests for ArXiv Agent with MMORE integration."""

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver


class FakeLLMWithTools(BaseChatModel):
    """Fake LLM that supports bind_tools() for testing."""

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs):
        """Initialize with a list of responses."""
        super().__init__(**kwargs)
        self._responses = responses or [AIMessage(content="Default response")]
        self._response_index = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
        else:
            response = self._responses[-1]  # Reuse last response

        generation = ChatGeneration(message=response)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model - just return self for testing."""
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "fake-llm-with-tools"


@pytest.fixture
def mock_arxiv_paper():
    """Mock ArXiv paper for testing."""
    mock = Mock()
    mock.entry_id = "http://arxiv.org/abs/1605.08386"
    mock.title = "Topology Optimization in Engineering"
    # Create proper mock authors with name attributes
    author1 = Mock()
    author1.name = "John Doe"
    author2 = Mock()
    author2.name = "Jane Smith"
    author3 = Mock()
    author3.name = "Bob Wilson"
    author4 = Mock()
    author4.name = "Alice Brown"
    mock.authors = [author1, author2, author3, author4]
    mock.summary = "This paper presents a comprehensive study of topology optimization methods in structural engineering applications. The research covers various optimization algorithms and their practical implementations in real-world scenarios."
    mock.published = datetime(2024, 1, 15)
    mock.categories = ["cs.CE", "math.OC"]
    mock.pdf_url = "http://arxiv.org/pdf/1605.08386"
    mock.doi = "10.1234/example.doi"
    mock.download_pdf.return_value = "/tmp/arxiv_papers/1605.08386.pdf"
    return mock


@pytest.fixture
def mock_arxiv_search(mock_arxiv_paper):
    """Mock ArXiv search for testing."""
    mock = Mock()
    mock.results.return_value = iter([mock_arxiv_paper])
    return mock


@pytest.fixture
def mock_mmore_client():
    """Mock MMORE client for testing."""
    mock = Mock()
    mock.health_check.return_value = True
    mock.upload_file.return_value = {"status": "success", "file_id": "arxiv_1605.08386"}
    mock.retrieve.return_value = [
        Document(
            page_content="Test content from paper about topology optimization",
            metadata={
                "source": "arxiv_1605.08386",
                "chunk_id": "chunk_1",
                "score": 0.95,
            },
        )
    ]
    mock.delete_file.return_value = {"status": "deleted"}
    return mock


@pytest.fixture
def mock_database():
    """Mock database manager for testing."""
    mock = Mock()
    mock.add_mmore_document.return_value = None
    mock.get_all_mmore_documents.return_value = [
        {
            "file_id": "arxiv_1605.08386",
            "file_name": "Topology Optimization in Engineering.pdf",
            "uploaded_at": datetime(2024, 1, 15, 10, 30),
        }
    ]
    return mock


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer for testing."""
    return MemorySaver()


class TestArXivAgentInitialization:
    """Test ArXiv agent initialization."""

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_arxiv_agent_initialization(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that ArXiv agent initializes correctly."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent(model_name="openai:gpt-4o")

        assert agent is not None
        assert agent.model_name == "openai:gpt-4o"
        mock_init_llm.assert_called_once_with("openai:gpt-4o", temperature=0.7)
        mock_mmore_cls.assert_called_once_with(base_url=None)
        mock_mmore_client.health_check.assert_called_once()

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_arxiv_agent_tools_created(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that ArXiv agent creates the expected tools."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()

        assert len(agent.tools) == 5
        tool_names = {tool.name for tool in agent.tools}
        assert "search_arxiv" in tool_names
        assert "get_arxiv_paper" in tool_names
        assert "download_and_analyze_paper" in tool_names
        assert "ask_about_papers" in tool_names
        assert "list_analyzed_papers" in tool_names

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_arxiv_agent_mmore_url(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that ArXiv agent accepts custom MMORE URL."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent(mmore_url="http://custom:8000")

        mock_mmore_cls.assert_called_once_with(base_url="http://custom:8000")


class TestArXivAgentSearchTool:
    """Test the search_arxiv tool."""

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_arxiv_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_arxiv_paper,
        mock_mmore_client,
    ):
        """Test successful ArXiv search."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        search_tool = agent.tools_by_name["search_arxiv"]

        result = search_tool.invoke({"query": "topology optimization"})

        assert "Found 1 papers" in result
        assert "Topology Optimization in Engineering" in result
        assert "1605.08386" in result
        assert "John Doe" in result
        mock_search_cls.assert_called_once()

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_arxiv_max_results(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_mmore_client,
    ):
        """Test ArXiv search with custom max results."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        search_tool = agent.tools_by_name["search_arxiv"]

        search_tool.invoke({"query": "machine learning", "max_results": 10})

        mock_search_cls.assert_called_once()
        call_args = mock_search_cls.call_args
        assert call_args.kwargs["max_results"] == 10

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_arxiv_no_results(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_mmore_client,
    ):
        """Test ArXiv search with no results."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search = Mock()
        mock_search.results.return_value = iter([])
        mock_search_cls.return_value = mock_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        search_tool = agent.tools_by_name["search_arxiv"]

        result = search_tool.invoke({"query": "nonexistent query xyz"})

        assert "No papers found" in result

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_arxiv_error_handling(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_mmore_client,
    ):
        """Test error handling in ArXiv search."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.side_effect = Exception("Network error")
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        search_tool = agent.tools_by_name["search_arxiv"]

        result = search_tool.invoke({"query": "test"})

        assert "Error searching ArXiv" in result


class TestArXivAgentGetPaperTool:
    """Test the get_arxiv_paper tool."""

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_get_arxiv_paper_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_arxiv_paper,
        mock_mmore_client,
    ):
        """Test successful paper retrieval."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        get_tool = agent.tools_by_name["get_arxiv_paper"]

        result = get_tool.invoke({"arxiv_id": "1605.08386"})

        assert "Topology Optimization in Engineering" in result
        assert "1605.08386" in result
        assert "John Doe" in result
        assert "Abstract:" in result
        mock_search_cls.assert_called_once()

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_get_arxiv_paper_with_prefix(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_mmore_client,
    ):
        """Test paper retrieval with arxiv: prefix."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        get_tool = agent.tools_by_name["get_arxiv_paper"]

        get_tool.invoke({"arxiv_id": "arxiv:1605.08386"})

        call_args = mock_search_cls.call_args
        assert call_args.kwargs["id_list"] == ["1605.08386"]

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_get_arxiv_paper_not_found(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_mmore_client,
    ):
        """Test paper retrieval when paper not found."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search = Mock()
        mock_search.results.return_value = iter([])
        mock_search_cls.return_value = mock_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        get_tool = agent.tools_by_name["get_arxiv_paper"]

        result = get_tool.invoke({"arxiv_id": "9999.99999"})

        assert "not found" in result


class TestArXivAgentDownloadAndAnalyzeTool:
    """Test the download_and_analyze_paper tool."""

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_download_and_analyze_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_arxiv_paper,
        mock_mmore_client,
        mock_database,
    ):
        """Test successful paper download and analysis."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent(db_manager=mock_database)

        download_tool = agent.tools_by_name["download_and_analyze_paper"]

        result = download_tool.invoke({"arxiv_id": "1605.08386"})

        assert "Successfully downloaded and analyzed" in result
        assert "Topology Optimization in Engineering" in result
        assert "1605.08386" in result
        mock_arxiv_paper.download_pdf.assert_called_once()
        mock_mmore_client.upload_file.assert_called_once()
        mock_database.add_mmore_document.assert_called_once()

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_download_and_analyze_paper_not_found(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_mmore_client,
    ):
        """Test download when paper not found."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search = Mock()
        mock_search.results.return_value = iter([])
        mock_search_cls.return_value = mock_search
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        download_tool = agent.tools_by_name["download_and_analyze_paper"]

        result = download_tool.invoke({"arxiv_id": "9999.99999"})

        assert "not found" in result

    @patch("src.tools.arxiv_tools.arxiv.Search")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_download_and_analyze_error_handling(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_search_cls,
        mock_arxiv_search,
        mock_arxiv_paper,
        mock_mmore_client,
    ):
        """Test error handling in download and analysis."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_search_cls.return_value = mock_arxiv_search
        mock_mmore_client.upload_file.side_effect = Exception("Upload failed")
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        download_tool = agent.tools_by_name["download_and_analyze_paper"]

        result = download_tool.invoke({"arxiv_id": "1605.08386"})

        assert "Error processing paper" in result


class TestArXivAgentAskPapersTool:
    """Test the ask_about_papers tool."""

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_ask_about_papers_success(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test successful paper query."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        ask_tool = agent.tools_by_name["ask_about_papers"]

        result = ask_tool.invoke({"query": "What is topology optimization?"})

        assert "Passage 1" in result
        assert "topology optimization" in result.lower()
        assert "relevance:" in result
        mock_mmore_client.retrieve.assert_called_once()

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_ask_about_papers_with_num_results(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test paper query with custom number of results."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        ask_tool = agent.tools_by_name["ask_about_papers"]

        ask_tool.invoke({"query": "test query", "num_results": 10})

        mock_mmore_client.retrieve.assert_called_once()
        call_args = mock_mmore_client.retrieve.call_args
        assert call_args.kwargs["max_matches"] == 10

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_ask_about_papers_no_results(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test paper query with no results."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.retrieve.return_value = []
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        ask_tool = agent.tools_by_name["ask_about_papers"]

        result = ask_tool.invoke({"query": "test"})

        assert "No relevant information found" in result

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_ask_about_papers_error_handling(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test error handling in paper query."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_client.retrieve.side_effect = Exception("Query failed")
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        ask_tool = agent.tools_by_name["ask_about_papers"]

        result = ask_tool.invoke({"query": "test"})

        assert "Error querying papers" in result


class TestArXivAgentListPapersTool:
    """Test the list_analyzed_papers tool."""

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_list_papers_success(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
        mock_database,
    ):
        """Test successful paper listing."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent(db_manager=mock_database)

        list_tool = agent.tools_by_name["list_analyzed_papers"]

        result = list_tool.invoke({})

        assert "ArXiv Papers in MMORE Knowledge Base" in result
        assert "1 paper(s)" in result
        assert "Topology Optimization in Engineering" in result
        assert "1605.08386" in result

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_list_papers_empty(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test listing when no papers exist."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        mock_db = Mock()
        mock_db.get_all_mmore_documents.return_value = []

        agent = ArXivAgent(db_manager=mock_db)

        list_tool = agent.tools_by_name["list_analyzed_papers"]

        result = list_tool.invoke({})

        assert "No ArXiv papers in MMORE knowledge base yet" in result
        assert "download_and_analyze_paper" in result

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_list_papers_error_handling(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test error handling in paper listing."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        mock_db = Mock()
        mock_db.get_all_mmore_documents.side_effect = Exception("Database error")

        agent = ArXivAgent(db_manager=mock_db)

        list_tool = agent.tools_by_name["list_analyzed_papers"]

        result = list_tool.invoke({})

        assert "Error listing papers" in result


class TestArXivAgentInvoke:
    """Test ArXiv agent invocation."""

    @patch("src.agents.base_agent.get_checkpointer")
    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_invoke_with_simple_query(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_get_checkpointer,
        mock_checkpointer,
        mock_mmore_client,
    ):
        """Test invoking ArXiv agent with a simple query."""
        from src.agents.arxiv_agent import ArXivAgent

        fake_llm = FakeLLMWithTools(
            responses=[
                AIMessage(content="Here are some papers on topology optimization.")
            ]
        )
        mock_init_llm.return_value = fake_llm
        mock_get_checkpointer.return_value = mock_checkpointer
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()
        state = {
            "messages": [HumanMessage(content="Find papers on topology optimization")]
        }
        config = {"configurable": {"thread_id": "test_thread"}}

        result = agent.invoke(state, config)

        assert result is not None
        assert "messages" in result


@pytest.mark.unit
class TestArXivAgentSystemPrompt:
    """Test ArXiv agent system prompt."""

    @patch("src.agents.arxiv_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_system_prompt_content(
        self,
        mock_init_llm,
        mock_mmore_cls,
        mock_mmore_client,
    ):
        """Test that system prompt is set correctly."""
        from src.agents.arxiv_agent import ArXivAgent

        mock_llm = FakeLLMWithTools()
        mock_init_llm.return_value = mock_llm
        mock_mmore_cls.return_value = mock_mmore_client

        agent = ArXivAgent()

        # Verify agent was created successfully with system prompt
        assert agent is not None
        system_prompt = agent._get_system_prompt()
        assert system_prompt is not None
        assert len(system_prompt) > 0
