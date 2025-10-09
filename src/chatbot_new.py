# Step 1: Define tools and model

import operator
from typing import Annotated, Literal, NotRequired, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import config

tavily_api_key = config.tavily_api_key

llm = init_chat_model(
    "openai:gpt-4.1",
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Step 2: Define state


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: NotRequired[int]


# Step 3: Define model node


def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or not"""

    messages: list[AnyMessage] = [
        SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
    ] + state["messages"]

    return {
        "messages": [llm_with_tools.invoke(messages)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Step 4: Define tool node


def tool_node(state: MessagesState) -> dict:
    """Performs the tool call"""

    result = []
    last_message = state["messages"][-1]

    # Only AIMessage has tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )

    return {"messages": result}


# Step 5: Define logic to determine whether to end


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    # Otherwise, we stop (reply to the user)
    return "__end__"


# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Invoke

initial_messages: list[AnyMessage] = [HumanMessage(content="Add 3 and 4.")]
initial_state: MessagesState = {"messages": initial_messages}
result = cast(MessagesState, agent.invoke(initial_state))  # type: ignore[arg-type]
for m in result["messages"]:
    m.pretty_print()
