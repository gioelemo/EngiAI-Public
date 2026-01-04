"""Test the compliance extraction logic with simulated tool message content."""

import ast
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Simulate what optimize_design returns (before stringification)
optimize_design_result = {
    "success": True,
    "problem_type": "beams2d",
    "design_shape": (50, 100),
    "optimized_design": [[0.1, 0.2], [0.3, 0.4]],  # Simplified for test
    "optimization_info": {},
    "initial_compliance": 120.5,
    "final_compliance": 65.3,
    "compliance_improvement": 45.8,
    "message": "Optimized beams2d design: compliance: 120.5→65.3 (45.8%)"
}

# This is what base_agent.py does at line 122
stringified_content = str(optimize_design_result)

print("=" * 60)
print("Testing compliance extraction")
print("=" * 60)

print(f"\n1. Original dict keys: {list(optimize_design_result.keys())}")
print(f"2. Stringified content type: {type(stringified_content)}")
print(f"3. Stringified content preview:\n{stringified_content[:200]}...\n")

# Test extraction (simulating what our function does)
print("4. Testing extraction methods:")

# Method 1: ast.literal_eval (should work!)
try:
    parsed = ast.literal_eval(stringified_content)
    print(f"   ✅ ast.literal_eval succeeded!")
    print(f"   Parsed type: {type(parsed)}")
    print(f"   Has final_compliance: {'final_compliance' in parsed}")

    if "final_compliance" in parsed:
        compliance_data = {
            "final_compliance": float(parsed["final_compliance"]),
            "initial_compliance": float(parsed.get("initial_compliance", 0)),
            "improvement": float(parsed.get("compliance_improvement", 0))
        }
        print(f"   ✅ Extracted compliance: {compliance_data}")
    else:
        print(f"   ❌ final_compliance not in parsed dict")
        print(f"   Available keys: {list(parsed.keys())}")

except (ValueError, SyntaxError) as e:
    print(f"   ❌ ast.literal_eval failed: {e}")

print("\n" + "=" * 60)
print("Conclusion: ast.literal_eval should work perfectly!")
print("If extraction is failing, the tool message isn't being found.")
print("=" * 60)
