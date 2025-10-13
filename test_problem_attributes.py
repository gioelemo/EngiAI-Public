"""
Test script demonstrating how to query problem attributes from the chatbot.

This shows how users can ask about design_space, objectives, and conditions
and get the information directly from the EngiBench problem object.
"""

from langchain_core.messages import HumanMessage

from src.agents.engineering_agent import EngineeringAgent


def test_problem_attributes():
    """Test querying problem attributes through the EngineeringAgent."""
    print("=" * 80)
    print("Testing Problem Attribute Queries")
    print("=" * 80)
    print()

    # Initialize the engineering agent
    agent = EngineeringAgent()

    # Test queries about problem attributes
    queries = [
        "What is the design space of the beams2d problem?",
        "What are the objectives for beam optimization?",
        "What conditions can I set for a beam problem?",
        "Tell me the design space shape and bounds",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 80}\n")

        state = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": f"test_attr_{i}"}}

        result = agent.invoke(state, config)

        # Print the response
        if result["messages"]:
            last_message = result["messages"][-1]
            print(f"Response:\n{last_message.content}\n")


if __name__ == "__main__":
    test_problem_attributes()
