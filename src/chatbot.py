from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from config import config


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


graph_builder = StateGraph(State)


@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


search_tool = TavilySearch(api_key=config.tavily_api_key, max_results=2)
tools = [search_tool, human_assistance]

llm = init_chat_model("openai:gpt-4.1")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)


tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

configurable = {"configurable": {"thread_id": "1"}}

print("🤖 LangGraph Chatbot is ready! Type 'exit' to quit.\n")

# Start an empty conversation
conversation = {
    "messages": [{"role": "user", "content": "Hello!"}],
}

while True:
    user_input = input("🧑 You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    # Append user message
    conversation["messages"].append({"role": "user", "content": user_input})

    # Stream events from LangGraph
    for event in graph.stream(conversation, configurable, stream_mode="values"): # type: ignore[arg-type]
        if "messages" in event:
            last_msg = event["messages"][-1]
            # Pretty print or raw content
            try:
                last_msg.pretty_print()
            except Exception:
                print("🤖 Bot:", last_msg.get("content", ""))

    # Retrieve and persist the new state (checkpointing)
    latest_state = None
    for state in graph.get_state_history(configurable): # type: ignore[arg-type]
        latest_state = state

    if latest_state:
        conversation = {"messages": latest_state.values["messages"]}
