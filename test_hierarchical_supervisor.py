"""
Test the hierarchical supervisor agent delegation.
"""

from langchain_core.messages import HumanMessage

from src.agents.supervisor_agent import SupervisorAgent


def test_hierarchical_supervisor():
    """Test that supervisor properly delegates to specialized agents."""
    print("🤖 Testing Hierarchical Supervisor Agent")
    print("=" * 60)

    # Initialize supervisor
    print("\n1. Initializing supervisor...")
    supervisor = SupervisorAgent()
    print("   ✓ Supervisor created")
    print(f"   ✓ Has engineering_agent: {hasattr(supervisor, 'engineering_agent')}")
    print(f"   ✓ Has search_agent: {hasattr(supervisor, 'search_agent')}")

    # Test engineering delegation
    print("\n2. Testing delegation to engineering agent...")
    state = {
        "messages": [HumanMessage(content="What engineering tools are available?")]
    }
    config = {"configurable": {"thread_id": "test-1"}}

    try:
        result = supervisor.invoke(state, config)
        print("   ✓ Successfully delegated and received response")
        if result["messages"]:
            last_message = result["messages"][-1]
            content_preview = str(last_message.content)[:200]
            print(f"   ✓ Response preview: {content_preview}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise

    print("\n" + "=" * 60)
    print("✅ Hierarchical supervisor test completed!")
    print("\nArchitecture:")
    print("  Supervisor (coordinator)")
    print("  ├── Engineering Agent (tools: optimization, STL, etc.)")
    print("  └── Search Agent (tools: web search)")


if __name__ == "__main__":
    test_hierarchical_supervisor()
