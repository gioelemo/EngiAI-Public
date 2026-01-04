"""Quick test to see message structure from optimize_design tool."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables before imports
os.environ["SKIP_MCP"] = "true"
os.environ["SKIP_MMORE"] = "true"

from langchain_core.messages import HumanMessage
from src.agents.supervisor_agent import SupervisorAgent

# Simple test prompt
prompt = """Design a 2D beam structure with the following constraints:
- Volume fraction: 35.0% (use only 35.0% of available material)
- Minimum feature size (rmin): 2.0
- Load condition: a uniformly distributed force
- Target: Minimize compliance (maximize stiffness)

The beam should be designed on a 50x100 grid."""

print("=" * 60)
print("Testing message structure from optimize_design")
print("=" * 60)

# Initialize supervisor
supervisor = SupervisorAgent(eval_mode=True)

# Create message
messages = [HumanMessage(content=prompt)]
state = {"messages": messages}

# Invoke
print("\n🤖 Invoking agent...")
result = supervisor.invoke(state, {"configurable": {"thread_id": "test_123"}})

# Analyze messages
print("\n📋 Analyzing messages...")
print(f"Total messages: {len(result['messages'])}")

for i, msg in enumerate(result['messages']):
    msg_type = type(msg).__name__
    print(f"\nMessage {i}: {msg_type}")

    # Check for tool messages
    if hasattr(msg, "tool_call_id"):
        tool_name = getattr(msg, "name", "unknown")
        print(f"  Tool name: {tool_name}")
        print(f"  Content type: {type(msg.content).__name__}")

        # If it's optimize_design, show the content
        if tool_name == "optimize_design":
            print(f"\n  ✅ FOUND optimize_design tool message!")
            print(f"  Content type: {type(msg.content)}")

            # Show content structure
            if isinstance(msg.content, dict):
                print(f"  Content is a DICT with keys: {list(msg.content.keys())}")
                if "final_compliance" in msg.content:
                    print(f"  ✅ final_compliance found: {msg.content['final_compliance']}")
                if "initial_compliance" in msg.content:
                    print(f"  ✅ initial_compliance found: {msg.content['initial_compliance']}")
            elif isinstance(msg.content, str):
                print(f"  Content is a STRING (length: {len(msg.content)})")
                print(f"  Content preview: {msg.content[:300]}")
                # Check if compliance is in string
                if "compliance" in msg.content.lower():
                    print(f"  ✅ 'compliance' found in string content")
            else:
                print(f"  Content is {type(msg.content)}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
