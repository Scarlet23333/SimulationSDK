"""
Production vs Simulation Mode Example - SimulationSDK

This example shows how to switch between simulation and production modes,
and demonstrates the agent-as-tool pattern for performance reuse.
Follows the patterns from test_refactored_sdk.py
"""

import os
from simulation_sdk import (
    simulation_tool,
    simulation_agent,
    ToolCategory,
    SimulationResponse,
    get_current_context,
    merge_performance_files
)


# Production-affecting tool
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "deployment_id": "deploy_sim_123",
            "app_name": "$app_name",
            "environment": "$environment",
            "status": "deployed",
            "url": "https://$app_name-$environment.example.com"
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_code=500,
        error_message="Deployment failed: insufficient resources"
    ),
    delay_ms=5000  # Simulate deployment time
)
def deploy_application(app_name: str, environment: str):
    """Deploy application to cloud environment - PRODUCTION ONLY"""
    print(f"[PRODUCTION] Deploying {app_name} to {environment}")
    # Real deployment logic would go here
    return {
        "deployment_id": f"deploy_prod_{environment}_456",
        "app_name": app_name,
        "environment": environment,
        "status": "deployed",
        "url": f"https://{app_name}.{environment}.real.com"
    }


# READ_ONLY tool for checking deployment status
@simulation_tool(
    category=ToolCategory.READ_ONLY,
    delay_ms=200
)
def check_deployment_status(deployment_id: str):
    """Check deployment status - safe to run anytime"""
    print(f"[READ_ONLY] Checking status for deployment: {deployment_id}")
    
    # Simulate checking deployment
    if deployment_id.startswith("deploy_sim"):
        return {"status": "simulated", "health": "n/a"}
    else:
        return {"status": "running", "health": "healthy"}


# Agent that performs deployment (with decorator for testing)
@simulation_agent(name="deployment_agent", delay_ms=1000)
def deployment_agent(app_name: str, run_tests: bool = True):
    """Agent that handles application deployment"""
    print(f"\n=== Deployment Agent ===")
    print(f"Deploying {app_name}, run_tests={run_tests}")
    
    if run_tests:
        print(f"Running tests for {app_name}...")
        # Simulate test execution
        import time
        time.sleep(0.2)
    
    # Deploy to staging first
    staging_result = deploy_application(app_name, "staging")
    print(f"Staging deployment: {staging_result['url']}")
    
    # Check staging status
    staging_status = check_deployment_status(staging_result["deployment_id"])
    print(f"Staging status: {staging_status}")
    
    # Deploy to production
    prod_result = deploy_application(app_name, "production")
    print(f"Production deployment: {prod_result['url']}")
    
    return {
        "status": "deployed",
        "app_name": app_name,
        "staging_url": staging_result["url"],
        "production_url": prod_result["url"],
        "deployments": [staging_result, prod_result]
    }


# After testing, create AGENT_TOOL wrapper
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="deployment_agent"
)
def deployment_agent_tool(app_name: str, run_tests: bool = True):
    """Wrapper to use deployment_agent as a tool with cached performance"""
    # In simulation mode, this loads and returns cached performance
    # In production mode, this executes the actual agent
    return deployment_agent(app_name, run_tests)


def demonstrate_mode_switching():
    """Show the difference between simulation and production modes"""
    
    print("=== Mode Switching Demonstration ===\n")
    
    print("Part 1: SIMULATION MODE (Safe Testing)")
    print("-" * 50)
    os.environ["SIMULATION_MODE"] = "true"
    
    # Deploy in simulation mode
    result = deployment_agent("my-app", run_tests=True)
    print(f"\nSimulation Result:")
    print(f"  Status: {result['status']}")
    print(f"  Staging URL: {result['staging_url']}")
    print(f"  Production URL: {result['production_url']}")
    print("  Note: No actual deployment occurred!")
    
    # Save performance
    merge_performance_files()
    
    print("\n\nPart 2: PRODUCTION MODE (Real Execution)")
    print("-" * 50)
    print("WARNING: In production mode, tools would actually execute!")
    print("(Keeping simulation on for safety in this demo)")
    
    # Normally you would do:
    # os.environ["SIMULATION_MODE"] = "false"
    # But we keep simulation on for demo safety
    
    print("\n\nPart 3: Agent as Tool (Performance Reuse)")
    print("-" * 50)
    
    # Using the agent as a tool - loads cached performance
    print("Using deployment_agent_tool (loads cached performance)...")
    tool_result = deployment_agent_tool("another-app", run_tests=False)
    
    print(f"\nTool Result:")
    print(f"  Status: {tool_result['status']}")
    print(f"  App: {tool_result['app_name']}")
    print("  This used cached performance - no re-execution!")


def demonstrate_error_injection():
    """Show how to test error handling"""
    
    print("\n\n=== Error Injection Demonstration ===\n")
    
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.3"  # 30% error rate
    
    print("Testing with 30% error rate (5 deployments):")
    print("-" * 50)
    
    successes = 0
    failures = 0
    
    for i in range(5):
        try:
            result = deploy_application(f"test-app-{i}", "test")
            print(f"Deploy {i+1}: SUCCESS - {result['status']}")
            successes += 1
        except Exception as e:
            print(f"Deploy {i+1}: FAILURE - Simulated deployment error")
            failures += 1
    
    print(f"\nResults: {successes} successes, {failures} failures")
    print("This helps test error handling without breaking production!")
    
    # Reset error rate
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"


def demonstrate_workflow_with_modes():
    """Show a complete workflow using different modes"""
    
    print("\n\n=== Workflow with Mode Control ===\n")
    
    context = get_current_context()
    context.start_workflow("deployment_workflow_001", "Multi-App Deployment")
    
    apps_to_deploy = ["frontend-app", "backend-api", "worker-service"]
    
    print(f"Deploying {len(apps_to_deploy)} applications...")
    print("-" * 50)
    
    deployment_results = []
    
    for app in apps_to_deploy:
        print(f"\nDeploying {app}...")
        
        # You could switch modes per app if needed
        # os.environ["SIMULATION_MODE"] = "false" if app == "frontend-app" else "true"
        
        result = deployment_agent_tool(app, run_tests=True)
        deployment_results.append(result)
        print(f"  ✓ {app} deployed successfully")
    
    # End workflow
    workflow_metrics = context.end_workflow(success=True, comment_score=9.0)
    
    if workflow_metrics:
        print(f"\n\nWorkflow Summary:")
        print(f"  Total Duration: {workflow_metrics.total_duration}ms")
        print(f"  Apps Deployed: {len(deployment_results)}")
        print(f"  Quality Score: {workflow_metrics.comment_score}/10")


if __name__ == "__main__":
    # Set up environment
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"
    
    print("=== Production vs Simulation Mode Example ===\n")
    
    # Run demonstrations
    demonstrate_mode_switching()
    demonstrate_error_injection()
    demonstrate_workflow_with_modes()
    
    print("\n\n=== Key Takeaways ===")
    print("1. Simulation mode: Safe testing with realistic responses")
    print("2. Production mode: Real tool execution")
    print("3. Agent-as-tool: Reuse performance without re-execution")
    print("4. Error injection: Test error handling safely")
    print("5. Environment control: Easy mode switching")
    
    print("\nExample completed!")