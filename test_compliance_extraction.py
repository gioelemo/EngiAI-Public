"""Quick test to check compliance extraction from recent evaluation."""
import logging
import sys
from pathlib import Path

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import weave
from benchmarks.shared.utils import extract_compliance_from_tool_messages

# Initialize weave
weave.init('engineer-assistant')

# Try to get a recent evaluation call
print("=" * 60)
print("CHECKING COMPLIANCE EXTRACTION")
print("=" * 60)

# Get calls to the evaluate function
calls = weave.calls(
    filter={'op_name': 'EngineeringAgent.predict'},
    limit=1
)

if calls:
    call = list(calls)[0]
    output = call.output

    if output and 'messages' in output:
        messages = output['messages']
        print(f"\nFound {len(messages)} messages in first evaluation")

        # Try to extract compliance
        compliance = extract_compliance_from_tool_messages(messages, example_id=0)

        if compliance:
            print(f"\n✅ SUCCESS! Extracted: {compliance}")
        else:
            print("\n❌ FAILED to extract compliance")
            print("\nLet's examine tool messages:")
            for i, msg in enumerate(messages):
                if hasattr(msg, 'tool_call_id'):
                    print(f"\n--- Tool Message {i} ---")
                    content = str(msg.content)
                    if len(content) > 500:
                        print(content[:500] + "...")
                    else:
                        print(content)
    else:
        print("No messages found in output")
else:
    print("No recent evaluation calls found")
    print("Please run an evaluation first:")
    print("  python benchmarks/evaluations/evaluate_agent.py --problem beams2d --samples 1")
