"""
Test script to demonstrate getting dataset information from the Engineering Agent.
"""

from langchain_core.messages import HumanMessage

from src.agents.engineering_agent import EngineeringAgent


def test_dataset_info():
    """Test the get_dataset_info tool through the EngineeringAgent."""
    print("=" * 80)
    print("Testing Dataset Information Retrieval")
    print("=" * 80)
    print()

    # Initialize the engineering agent
    agent = EngineeringAgent()

    # Test query about dataset information
    queries = [
        "What information is available about the dataset?",
        "Tell me about the EngiBench dataset for beam problems",
        "How many samples are in the dataset?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 80}\n")

        state = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": "test_dataset"}}

        result = agent.invoke(state, config)

        # Print the response
        if result["messages"]:
            last_message = result["messages"][-1]
            print(f"Response:\n{last_message.content}\n")


if __name__ == "__main__":
    test_dataset_info()
