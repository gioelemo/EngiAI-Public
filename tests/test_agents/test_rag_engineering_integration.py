"""Tests for RAG tools integration with the Engineering Agent.

Verifies that:
1. create_rag_tools() produces the correct tools from an MMOREClient
2. EngineeringAgent includes RAG tools alongside EngiBench tools
3. RAG tools work correctly when invoked through the EngineeringAgent
4. EngineeringAgent gracefully handles SKIP_MMORE=true
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from tests.test_agents.test_rag_agent import FakeLLMWithTools

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_mmore_client():
    """Mock MMORE client for testing."""
    mock = Mock()
    mock.health_check.return_value = True
    mock.retrieve.return_value = [
        Document(
            page_content="Topology optimization content",
            metadata={"source": "eng_paper.pdf", "chunk_id": "1", "score": 0.92},
        )
    ]
    mock.upload_file.return_value = {"status": "success", "fileId": "test_id"}
    mock.delete_file.return_value = {"status": "success"}
    mock.list_files.return_value = ["eng_paper.pdf", "design_guide.pdf"]
    return mock


# ============================================================================
# RAG TOOLS FACTORY TESTS
# ============================================================================


class TestCreateRagTools:
    """Test the create_rag_tools factory function."""

    @pytest.mark.unit
    def test_creates_all_five_tools(self, mock_mmore_client):
        """Test that create_rag_tools returns exactly 5 tools."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)

        assert len(tools) == 5

    @pytest.mark.unit
    def test_tool_names(self, mock_mmore_client):
        """Test that all expected tool names are present."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        tool_names = {t.name for t in tools}

        assert tool_names == {
            "search_documents",
            "add_document",
            "add_url_to_knowledge_base",
            "list_documents",
            "delete_document",
        }

    @pytest.mark.unit
    def test_tools_have_descriptions(self, mock_mmore_client):
        """Test that all tools have non-empty descriptions."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)

        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"
            assert len(tool.description) > 10

    @pytest.mark.unit
    def test_search_tool_calls_retrieve(self, mock_mmore_client):
        """Test that search_documents calls mmore_client.retrieve."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "topology optimization"})

        mock_mmore_client.retrieve.assert_called_once_with(
            query="topology optimization", max_matches=5, min_similarity=0.3
        )
        assert "eng_paper.pdf" in result
        assert "0.92" in result

    @pytest.mark.unit
    def test_search_tool_no_results(self, mock_mmore_client):
        """Test search_documents when no documents are found."""
        from src.tools.rag_tools import create_rag_tools

        mock_mmore_client.retrieve.return_value = []
        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "nonexistent topic"})

        assert "No relevant documents found" in result

    @pytest.mark.unit
    def test_search_tool_error_handling(self, mock_mmore_client):
        """Test search_documents handles exceptions gracefully."""
        from src.tools.rag_tools import create_rag_tools

        mock_mmore_client.retrieve.side_effect = Exception("Connection refused")
        tools = create_rag_tools(mock_mmore_client)
        search_tool = next(t for t in tools if t.name == "search_documents")

        result = search_tool.invoke({"query": "test"})

        assert "Error searching documents" in result

    @pytest.mark.unit
    def test_list_documents_tool(self, mock_mmore_client):
        """Test list_documents returns file list."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        list_tool = next(t for t in tools if t.name == "list_documents")

        result = list_tool.invoke({})

        assert "2 document" in result
        assert "eng_paper.pdf" in result
        assert "design_guide.pdf" in result

    @pytest.mark.unit
    def test_delete_document_tool(self, mock_mmore_client):
        """Test delete_document calls mmore_client.delete_file."""
        from src.tools.rag_tools import create_rag_tools

        tools = create_rag_tools(mock_mmore_client)
        delete_tool = next(t for t in tools if t.name == "delete_document")

        result = delete_tool.invoke({"file_id": "eng_paper.pdf"})

        mock_mmore_client.delete_file.assert_called_once_with("eng_paper.pdf")
        assert "Deleted" in result


# ============================================================================
# ENGINEERING AGENT WITH RAG TOOLS
# ============================================================================


class TestEngineeringAgentRagIntegration:
    """Test that EngineeringAgent correctly includes RAG tools."""

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_engineering_agent_has_rag_tools(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that EngineeringAgent includes all 5 RAG tools."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()

        tool_names = {t.name for t in agent.tools}
        rag_tools = {
            "search_documents",
            "add_document",
            "add_url_to_knowledge_base",
            "list_documents",
            "delete_document",
        }
        assert rag_tools.issubset(tool_names), (
            f"Missing RAG tools: {rag_tools - tool_names}"
        )

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_engineering_agent_keeps_engibench_tools(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that EngineeringAgent still has EngiBench tools alongside RAG tools."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()

        tool_names = {t.name for t in agent.tools}
        engibench_tools = {
            "create_problem",
            "simulate_design",
            "optimize_design",
            "render_design",
            "convert_design_to_stl",
        }
        assert engibench_tools.issubset(tool_names), (
            f"Missing EngiBench tools: {engibench_tools - tool_names}"
        )

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_engineering_agent_total_tool_count(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that EngineeringAgent has correct total number of tools."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()

        # 9 engineering tools + 5 RAG tools = 14
        assert len(agent.tools) == 14

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_search_documents_via_engineering_agent(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that search_documents works when invoked through EngineeringAgent."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()
        search_tool = agent.tools_by_name["search_documents"]

        result = search_tool.invoke({"query": "beam optimization parameters"})

        mock_mmore_client.retrieve.assert_called_once()
        assert "eng_paper.pdf" in result

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_engineering_agent_mmore_client_attribute(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that EngineeringAgent exposes mmore_client attribute."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()

        assert agent.mmore_client is not None
        assert agent.mmore_client is mock_mmore_client

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_engineering_agent_custom_mmore_url(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that EngineeringAgent passes mmore_url to MMOREClient."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        EngineeringAgent(mmore_url="http://custom:9000")

        mock_mmore_cls.assert_called_once_with(base_url="http://custom:9000")


# ============================================================================
# SKIP_MMORE BEHAVIOR
# ============================================================================


class TestEngineeringAgentSkipMmore:
    """Test EngineeringAgent behavior when SKIP_MMORE=true."""

    @pytest.mark.unit
    @patch.dict("os.environ", {"SKIP_MMORE": "true"})
    @patch("src.agents.base_agent.init_chat_model")
    def test_no_rag_tools_when_skipped(self, mock_init_llm):
        """Test that no RAG tools are added when SKIP_MMORE=true."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()

        agent = EngineeringAgent()

        tool_names = {t.name for t in agent.tools}
        assert "search_documents" not in tool_names
        assert "list_documents" not in tool_names

    @pytest.mark.unit
    @patch.dict("os.environ", {"SKIP_MMORE": "true"})
    @patch("src.agents.base_agent.init_chat_model")
    def test_mmore_client_is_none_when_skipped(self, mock_init_llm):
        """Test that mmore_client is None when SKIP_MMORE=true."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()

        agent = EngineeringAgent()

        assert agent.mmore_client is None

    @pytest.mark.unit
    @patch.dict("os.environ", {"SKIP_MMORE": "true"})
    @patch("src.agents.base_agent.init_chat_model")
    def test_engibench_tools_still_present_when_skipped(self, mock_init_llm):
        """Test that EngiBench tools are still available when MMORE is skipped."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()

        agent = EngineeringAgent()

        tool_names = {t.name for t in agent.tools}
        assert "create_problem" in tool_names
        assert "simulate_design" in tool_names
        assert "optimize_design" in tool_names
        # 9 engineering tools, 0 RAG tools
        assert len(agent.tools) == 9


# ============================================================================
# SYSTEM PROMPT TESTS
# ============================================================================


class TestEngineeringAgentSystemPrompt:
    """Test that the engineering agent system prompt includes RAG documentation."""

    @pytest.mark.unit
    @patch("src.utils.prompts.config")
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_prompt_mentions_rag_tools(
        self, mock_init_llm, mock_mmore_cls, mock_config, mock_mmore_client
    ):
        """Test that system prompt documents the RAG tools."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client
        # Mock config.mmore_enabled to return True (simulates MMORE service running)
        mock_config.mmore_enabled = True

        agent = EngineeringAgent()
        prompt = agent._get_system_prompt()

        assert "search_documents" in prompt
        assert "list_documents" in prompt
        assert "Knowledge Base" in prompt

    @pytest.mark.unit
    @patch("src.agents.engineering_agent.MMOREClient")
    @patch("src.agents.base_agent.init_chat_model")
    def test_prompt_still_has_engibench_docs(
        self, mock_init_llm, mock_mmore_cls, mock_mmore_client
    ):
        """Test that system prompt still documents EngiBench tools."""
        from src.agents.engineering_agent import EngineeringAgent

        mock_init_llm.return_value = FakeLLMWithTools()
        mock_mmore_cls.return_value = mock_mmore_client

        agent = EngineeringAgent()
        prompt = agent._get_system_prompt()

        assert "EngiBench" in prompt
        assert "EngiOpt" in prompt
        assert "create_problem" in prompt
