"""
Test the enhanced simulation_tool decorator with AGENT_TOOL category.

This demonstrates the smart behavior where:
- If performance exists, it uses it
- If performance doesn't exist, it executes the function as simulation_agent first
"""

import os
from simulation_sdk import (
    simulation_tool,
    ToolCategory,
    merge_performance_files,
    PerformanceManager
)

# Set simulation mode
os.environ["SIMULATION_MODE"] = "true"


# Test with a completely new agent name
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="report_generator_v2",
    description="Generates analytical reports"
)
def generate_report(topic: str, data: dict) -> dict:
    """Generate an analytical report based on the provided data."""
    print(f"[ACTUAL EXECUTION] Generating report for: {topic}")
    
    # Simulate report generation
    report = {
        "title": f"Report on {topic}",
        "sections": [
            {"name": "Executive Summary", "content": f"Analysis of {topic} data"},
            {"name": "Data Overview", "content": f"Total items: {len(data)}"},
            {"name": "Key Findings", "content": "Important patterns identified"},
            {"name": "Recommendations", "content": "Based on analysis, we recommend..."}
        ],
        "metadata": {
            "generated_by": "report_generator_v2",
            "data_points": len(data)
        }
    }
    return report


def test_smart_decorator():
    """Test the smart decorator behavior."""
    print("=== Testing Smart Decorator for AGENT_TOOL ===\n")
    
    # Check if performance exists
    pm = PerformanceManager()
    existing = pm.load_agent_performance("report_generator_v2")
    print(f"1. Initial state - Performance exists: {existing is not None}")
    
    # First call - should execute the function and save performance
    print("\n2. First call to generate_report:")
    print("-" * 50)
    result1 = generate_report("Q4 Sales", {"sales": [100, 200, 300]})
    print(f"Result type: {type(result1)}")
    print(f"Result keys: {result1.keys() if isinstance(result1, dict) else 'Not a dict'}")
    
    # Check temp files before merge
    temp_files = list(pm.temp_dir.glob("*.json"))
    print(f"\n3. Temp files created: {len(temp_files)}")
    
    # Merge temp files
    print("\n4. Merging temp files...")
    merge_performance_files()
    
    # Check if performance now exists
    existing_after = pm.load_agent_performance("report_generator_v2")
    print(f"Performance exists after merge: {existing_after is not None}")
    
    # Second call - should use saved performance
    print("\n5. Second call to generate_report:")
    print("-" * 50)
    result2 = generate_report("Customer Feedback", {"feedback": ["good", "excellent"]})
    print(f"Result type: {type(result2)}")
    if isinstance(result2, dict):
        print(f"Result title: {result2.get('title', 'No title')}")
    
    # Verify the results match
    print("\n6. Verification:")
    print(f"Both results are dicts: {isinstance(result1, dict) and isinstance(result2, dict)}")
    if isinstance(result1, dict) and isinstance(result2, dict):
        print(f"Results are identical: {result1 == result2}")
        if result1 != result2:
            print("  Note: Results differ because saved performance captures the first execution")


def test_read_only_tool():
    """Test that READ_ONLY tools always execute."""
    print("\n\n=== Testing READ_ONLY Tool Behavior ===\n")
    
    @simulation_tool(
        category=ToolCategory.READ_ONLY,
        description="Reads current timestamp"
    )
    def get_timestamp():
        import time
        return {"timestamp": time.time()}
    
    # Multiple calls should give different timestamps
    result1 = get_timestamp()
    import time
    time.sleep(0.1)
    result2 = get_timestamp()
    
    print(f"First timestamp: {result1['timestamp']}")
    print(f"Second timestamp: {result2['timestamp']}")
    print(f"Different timestamps (as expected): {result1['timestamp'] != result2['timestamp']}")


if __name__ == "__main__":
    test_smart_decorator()
    test_read_only_tool()
    print("\n=== All tests completed ===")