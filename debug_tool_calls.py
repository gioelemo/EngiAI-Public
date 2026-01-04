"""Debug script to see what tools are being called."""
import os
import sys
from pathlib import Path

os.environ["SKIP_MCP"] = "true"
os.environ["SKIP_MMORE"] = "true"

sys.path.insert(0, str(Path.cwd()))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agents.supervisor_agent import SupervisorAgent

prompt = """Design a 2D beam structure with the following constraints:
- Volume fraction: 23.8% (use only 23.8% of available material)
- Minimum feature size (rmin): 3.5
- Load condition: a uniformly distributed force
- Target: Minimize compliance (maximize stiffness)

The beam should be designed on a 50x100 grid."""

print("=" * 60)
print("Analyzing agent tool calls")
print("=" * 60)

supervisor = SupervisorAgent(eval_mode=True)
messages = [HumanMessage(content=prompt)]
result = supervisor.invoke({"messages": messages}, {"configurable": {"thread_id": "debug_123"}})

print(f"\nTotal messages: {len(result['messages'])}\n")

for i, msg in enumerate(result["messages"]):
    msg_type = type(msg).__name__
    print(f"Message {i}: {msg_type}")

    # Check for tool calls (AIMessage)
    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → Tool call: {tc['name']}")
            print(f"     Args: {tc['args']}")

    # Check for tool results (ToolMessage)
    if isinstance(msg, ToolMessage):
        tool_name = getattr(msg, "name", "unknown")
        print(f"  ← Tool result from: {tool_name}")
        content_str = str(msg.content)
        print(f"     Content length: {len(content_str)} chars")
        print(f"     Content preview: {content_str[:150]}...")

        # Check if this is optimize_design
        if tool_name == "optimize_design":
            print(f"     ✅ THIS IS optimize_design!")
            if "compliance" in content_str.lower():
                print(f"     ✅ Contains 'compliance'")
            else:
                print(f"     ❌ Does NOT contain 'compliance'")

print("\n" + "=" * 60)
