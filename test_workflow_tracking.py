#!/usr/bin/env python3
"""
Test script to demonstrate workflow tracking functionality.
"""

import os
import time
from simulation_sdk import (
    simulation_tool, 
    simulation_agent,
    ToolCategory,
    SimulationResponse,
    get_current_context,
    save_workflow_metrics,
    merge_performance_files
)

# Enable simulation mode
os.environ["SIMULATION_MODE"] = "true"

# Define some test tools
@simulation_tool(
    category=ToolCategory.READ_ONLY,
    delay_ms=100,
    description="Read file contents"
)
def read_file(filename: str) -> str:
    """Read contents of a file."""
    return f"Contents of {filename}"


@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    delay_ms=200,
    success_template=SimulationResponse(
        success=True,
        response_data={"message": "File {filename} written successfully"}
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_message="Failed to write file"
    ),
    description="Write file contents"
)
def write_file(filename: str, content: str) -> dict:
    """Write contents to a file."""
    return {"message": f"Wrote {len(content)} bytes to {filename}"}


# Define test agents
@simulation_agent(name="code_analyzer", delay_ms=500)
def analyze_code(file_path: str) -> dict:
    """Analyze code file and return insights."""
    # Simulate using tools
    content = read_file(file_path)
    
    return {
        "file": file_path,
        "lines": 42,
        "complexity": "medium",
        "suggestions": ["Add more comments", "Refactor long functions"]
    }


@simulation_agent(name="code_improver", delay_ms=800)
def improve_code(file_path: str, suggestions: list) -> dict:
    """Improve code based on suggestions."""
    # Simulate reading and writing
    original = read_file(file_path)
    
    # Apply improvements
    improved_content = f"{original}\n# Improvements applied"
    result = write_file(file_path, improved_content)
    
    return {
        "file": file_path,
        "improvements_applied": len(suggestions),
        "status": "success"
    }


def test_workflow_tracking():
    """Test the workflow tracking functionality."""
    print("Testing workflow tracking...")
    
    # Get current context
    context = get_current_context()
    
    # Start a workflow
    workflow_id = "refactor_workflow_001"
    workflow_name = "Code Refactoring Workflow"
    context.start_workflow(workflow_id, workflow_name)
    
    print(f"Started workflow: {workflow_name}")
    
    try:
        # Execute agents as part of workflow
        print("\n1. Analyzing code...")
        analysis = analyze_code("example.py")
        print(f"   Analysis complete: {analysis}")
        
        print("\n2. Improving code...")
        improvement = improve_code("example.py", analysis["suggestions"])
        print(f"   Improvement complete: {improvement}")
        
        # End workflow successfully
        workflow_metrics = context.end_workflow(success=True, comment_score=9)
        
        if workflow_metrics:
            print(f"\nWorkflow completed successfully!")
            print(f"Total tasks: {len(workflow_metrics.tasks)}")
            print(f"Total tokens: {workflow_metrics.total_tokens}")
            print(f"Total duration: {workflow_metrics.total_duration}ms")
            print(f"Total cost: ${workflow_metrics.total_cost:.4f}")
            
            # Save workflow metrics
            saved_path = save_workflow_metrics(workflow_metrics)
            print(f"\nWorkflow metrics saved to: {saved_path}")
        
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        # End workflow with failure
        workflow_metrics = context.end_workflow(success=False, comment_score=3)
        if workflow_metrics:
            save_workflow_metrics(workflow_metrics)


def test_nested_workflow():
    """Test nested agent calls within a workflow."""
    print("\n\nTesting nested workflow...")
    
    context = get_current_context()
    
    # Start workflow
    context.start_workflow("nested_workflow_001", "Nested Agent Workflow")
    
    # Define a master agent that calls other agents
    @simulation_agent(name="master_agent", delay_ms=300)
    def master_refactor(file_list: list) -> dict:
        """Master agent that coordinates refactoring multiple files."""
        results = []
        
        for file_path in file_list:
            # Analyze each file
            analysis = analyze_code(file_path)
            
            # Improve based on analysis
            improvement = improve_code(file_path, analysis["suggestions"])
            
            results.append({
                "file": file_path,
                "analysis": analysis,
                "improvement": improvement
            })
        
        return {
            "files_processed": len(file_list),
            "results": results
        }
    
    # Execute master agent
    print("\nExecuting master agent...")
    result = master_refactor(["file1.py", "file2.py", "file3.py"])
    print(f"Master agent complete: {result['files_processed']} files processed")
    
    # End workflow
    workflow_metrics = context.end_workflow(success=True, comment_score=10)
    
    if workflow_metrics:
        print(f"\nNested workflow completed!")
        print(f"Total tasks: {len(workflow_metrics.tasks)}")
        print(f"Total tokens: {workflow_metrics.total_tokens}")
        
        # Save metrics
        saved_path = save_workflow_metrics(workflow_metrics)
        print(f"Metrics saved to: {saved_path}")


if __name__ == "__main__":
    # Run tests
    test_workflow_tracking()
    test_nested_workflow()
    merge_performance_files()
    print("\n\nAll tests completed!")