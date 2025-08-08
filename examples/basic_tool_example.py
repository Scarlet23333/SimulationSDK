"""
Basic Tool Example - SimulationSDK

This example shows how to create and use simulated tools for safe testing.
Follows the patterns from test_refactored_sdk.py
"""

import os
from simulation_sdk import (
    merge_performance_files,
    simulation_tool, 
    simulation_agent,
    ToolCategory, 
    SimulationResponse,
    test_connectivity
)
from simulation_sdk.decorators import simulation_agent


# Example 1: READ_ONLY tool (always executes)
@simulation_tool(
    category=ToolCategory.READ_ONLY,
    delay_ms=100
)
def search_database(query: str) -> dict:
    """Search for items in the database - safe to run anytime"""
    print(f"[READ_ONLY] Searching database for: {query}")
    
    # Simulate database search
    if query == "python":
        return {
            "count": 3,
            "items": ["Python basics", "Advanced Python", "Python cookbook"]
        }
    return {"count": 0, "items": []}


# Example 2: PRODUCTION_AFFECTING tool (simulated in test mode)
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "email_id": "email_12345",
            "recipient": "$recipient",
            "subject": "$subject",
            "status": "sent"
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_code=400,
        error_message="Invalid recipient email address"
    ),
    delay_ms=500
)
def send_email(recipient: str, subject: str, body: str) -> dict:
    """Send an email - only runs in production mode"""
    print(f"[PRODUCTION] Sending email to {recipient}")
    # Real email implementation here
    return {
        "email_id": "prod_email_789",
        "recipient": recipient,
        "subject": subject,
        "status": "sent"
    }


# Example 3: Simple agent for testing
@simulation_agent(name="email_notification_agent_of_basic_tool_example", delay_ms=800)
def email_notification_agent(user_email: str, search_query: str):
    """Example agent that searches and sends results via email"""
    print(f"\n=== Email Notification Agent ===")
    
    # Search for items (READ_ONLY - always executes)
    search_results = search_database(search_query)
    print(f"Found {search_results['count']} items")
    
    # Send email notification (PRODUCTION_AFFECTING - simulated in test mode)
    if search_results['count'] > 0:
        email_result = send_email(
            recipient=user_email,
            subject=f"Search results for: {search_query}",
            body=f"We found {search_results['count']} items: {', '.join(search_results['items'])}"
        )
        print(f"Email result: {email_result}")
        return {
            "status": "success",
            "items_found": search_results['count'],
            "email_sent": True,
            "email_id": email_result.get("email_id")
        }
    else:
        return {
            "status": "no_results",
            "items_found": 0,
            "email_sent": False
        }


def demonstrate_modes():
    """Show the difference between simulation and production modes"""
    
    print("=== Mode Demonstration ===\n")
    
    # Test in simulation mode (default)
    print("1. SIMULATION MODE (SIMULATION_MODE=true)")
    print("-" * 40)
    os.environ["SIMULATION_MODE"] = "true"
    
    result = email_notification_agent("user@example.com", "python")
    print(f"Agent result: {result}")
    merge_performance_files()
    
    # Test in production mode
    print("\n2. PRODUCTION MODE (SIMULATION_MODE=false)")
    print("-" * 40)
    os.environ["SIMULATION_MODE"] = "false"
    
    result = email_notification_agent("user@example.com", "python")
    print(f"Agent result: {result}")
    
    # Reset to simulation mode
    os.environ["SIMULATION_MODE"] = "true"


def demonstrate_error_injection():
    """Show how error injection works in simulation"""
    
    print("\n\n=== Error Injection Demonstration ===\n")
    
    # Set high error rate
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.5"  # 50% error rate
    
    print("Testing with 50% error rate (5 attempts):")
    print("-" * 40)
    
    successes = 0
    failures = 0
    
    for i in range(5):
        try:
            result = send_email(
                recipient=f"user{i}@example.com",
                subject="Test",
                body="Test email"
            )
            print(f"Attempt {i+1}: SUCCESS - {result.get('status', 'sent')}")
            successes += 1
        except Exception as e:
            print(f"Attempt {i+1}: FAILURE - Simulated error")
            failures += 1
    
    print(f"\nResults: {successes} successes, {failures} failures")
    
    # Reset error rate
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"


if __name__ == "__main__":
    # Set up environment
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"
    
    print("=== Basic Tool Example ===\n")
    
    # Run demonstrations
    demonstrate_modes()
    demonstrate_error_injection()
    
    # Test connectivity
    print("\n\n=== Connectivity Test ===")
    print("-" * 40)
    connectivity = test_connectivity()
    print(f"API connectivity: {connectivity}")
    
    print("\n\nExample completed!")