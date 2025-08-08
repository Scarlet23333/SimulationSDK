#!/usr/bin/env python3
"""
Test file to validate the refactored SimulationSDK implementation.
"""

import os
import time
from simulation_sdk import (
    simulation_tool,
    simulation_agent,
    ToolCategory,
    test_connectivity,
    merge_performance_files,
    SimulationResponse,
)


# Example 1: PRODUCTION_AFFECTING tool
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "presentationId": "sim_pres_id_123",
            "status": "created",
            "title": "$title",
            "slides": "$num_slides"
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_code=403,
        error_message="Permission denied"
    ),
    delay_ms=250
)
def create_google_slides(title: str, num_slides: int):
    """Create a Google Slides presentation."""
    # This would normally call the real API
    print(f"Creating Google Slides presentation: {title} with {num_slides} slides")
    return {
        "presentationId": f"real_pres_123",
        "status": "created", 
        "title": title,
        "slides": num_slides
    }


# Example 2: READ_ONLY tool
@simulation_tool(category=ToolCategory.READ_ONLY)
def get_google_slides(presentation_id: str):
    """Get a Google Slides presentation."""
    # This always executes, even in simulation mode
    if presentation_id.startswith("sim_"):
        return {"error": "not_found", "id": presentation_id}
    return {"id": presentation_id, "title": "Test Presentation"}


# Example 3: Agent marked for testing
@simulation_agent(name="email_writer")
def email_writer_agent(recipient: str, subject: str):
    """Write and send an email."""
    print(f"Agent: Writing email to {recipient} with subject: {subject}")
    
    # Simulate some work
    time.sleep(0.1)
    
    # Use a tool
    slides = create_google_slides(f"Email attachment for {recipient}", 3)
    
    return {
        "email_id": f"email_{int(time.time())}",
        "recipient": recipient,
        "subject": subject,
        "attachment": slides.get("presentationId"),
        "status": "sent"
    }

# Example 4: AGENT_TOOL wrapper (after testing)
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="email_writer",
    delay_range=(1000, 2000)
)
def email_writer_agent_tool(recipient: str, subject: str):
    """Write and send an email."""
    return email_writer_agent(recipient, subject)

def test_basic_functionality():
    """Test basic SDK functionality."""
    print("=== Testing Refactored SimulationSDK ===\n")
    
    # Test 1: Production mode
    print("1. PRODUCTION MODE")
    print("-" * 40)
    os.environ["SIMULATION_MODE"] = "false"
    
    result = create_google_slides("Production Test", 5)
    print(f"Production result: {result}")
    
    # Test 2: Simulation mode
    print("\n2. SIMULATION MODE")
    print("-" * 40)
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"
    
    result = create_google_slides("Simulation Test", 3)
    print(f"Simulated result: {result}")
    
    # Test 3: READ_ONLY always executes
    print("\n3. READ_ONLY TOOL")
    print("-" * 40)
    
    result = get_google_slides("sim_pres_12345")
    print(f"Read-only result (sim ID): {result}")
    
    result = get_google_slides("real_pres_456")
    print(f"Read-only result (real ID): {result}")
    
    # Test 4: Agent execution
    print("\n4. AGENT EXECUTION")
    print("-" * 40)
    result = email_writer_agent("test@example.com", "Test Email")
    print(f"Agent result in simulation mode: {result}")


    os.environ["SIMULATION_MODE"] = "false"
    result = email_writer_agent("test@example.com", "Test Email")
    print(f"Agent result in production mode: {result}")
    
    os.environ["SIMULATION_MODE"] = "true"
    
    try:
        merge_performance_files()
        print("Performance files merged successfully")
    except Exception as e:
        print(f"Performance merge info: {e}")    
    
    result = email_writer_agent_tool("test@example.com", "Test Email")
    print(f"Agent tool result: {result}")

    # Test 5: Connectivity
    print("\n5. CONNECTIVITY TEST")
    print("-" * 40)
    
    connectivity = test_connectivity()
    print(f"Connectivity results: {connectivity}")
    
    # Test 6: Performance merge (would normally be CLI)
    print("\n6. PERFORMANCE MANAGEMENT")
    print("-" * 40)
    

    
    print("\n" + "=" * 50)
    print("All tests completed!")


def test_error_injection():
    """Test error injection in simulation mode."""
    print("\n=== Testing Error Injection ===\n")
    
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.5"  # 50% error rate
    
    print("Testing with 50% error rate (4 calls):")
    successes = 0
    failures = 0
    
    for i in range(4):
        try:
            result = create_google_slides(f"Test {i+1}", 2)
            print(f"  Call {i+1}: SUCCESS - {result.get('status', 'unknown')}")
            successes += 1
        except Exception as e:
            print(f"  Call {i+1}: FAILURE - Simulated error")
            failures += 1
    
    print(f"\nResults: {successes} successes, {failures} failures")


if __name__ == "__main__":
    test_basic_functionality()
    test_error_injection()