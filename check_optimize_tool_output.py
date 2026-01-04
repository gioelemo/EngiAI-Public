"""Check what optimize_design tool actually returns."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tools.engibench import optimize_design

print("=" * 60)
print("TESTING optimize_design TOOL OUTPUT FORMAT")
print("=" * 60)

# Test the optimize_design tool with a simple config
print("\nCalling optimize_design with beams2d...")
result = optimize_design.invoke({
    "problem_type": "beams2d",
    "config": {"volfrac": 0.25, "rmin": 3.5, "forcedist": 1.0},
    "seed": 42
})

print("\n✅ Tool returned successfully!")
print("\nType of result:", type(result))
print("\nResult content (first 500 chars):")
print(str(result)[:500])
print("...")

# Check if it's a string or dict
if isinstance(result, str):
    print("\n⚠️  Result is a STRING")
    print("Checking if 'compliance' is in the string:", "compliance" in result.lower())
    print("Checking if 'final_compliance' is in the string:", "final_compliance" in result.lower())
elif isinstance(result, dict):
    print("\n✅ Result is a DICT")
    print("Keys:", list(result.keys()))
    if "final_compliance" in result:
        print(f"final_compliance found: {result['final_compliance']}")
    else:
        print("⚠️  'final_compliance' NOT in dict keys")
else:
    print(f"\n❌ Unexpected type: {type(result)}")
