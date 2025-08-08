"""
Smart Agent and Workflow Example - SimulationSDK

This example demonstrates the smart agent workflow pattern. 
Decorate agents (llm-based nodes in LangGraph) with @simulation_tool(category=ToolCategory.AGENT_TOOL) to automatically save their performance (if it did not exist) and use them as tools.
"""

import os
import time
from simulation_sdk import (
    simulation_tool,
    ToolCategory,
    SimulationResponse,
    get_current_context,
    save_workflow_metrics,
    merge_performance_files
)


# Step 1: Define tools that agents will use
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "presentationId": "sim_pres_$timestamp",
            "title": "$title",
            "slides": "$num_slides",
            "url": "https://docs.google.com/presentation/d/sim_pres_$timestamp"
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_code=403,
        error_message="Insufficient permissions to create presentation"
    ),
    delay_ms=2000
)
def create_google_slides(title: str, num_slides: int):
    """Create a Google Slides presentation"""
    print(f"[PRODUCTION] Creating presentation: {title}")
    return {
        "presentationId": f"real_pres_{int(time.time())}",
        "title": title,
        "slides": num_slides,
        "url": f"https://docs.google.com/presentation/d/real_pres_{int(time.time())}"
    }


@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "slide_id": "slide_$slide_num",
            "status": "updated",
            "content_preview": "$content[:50]..."
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_code=404,
        error_message="Slide not found"
    ),
    delay_ms=500
)
def add_slide_content(presentation_id: str, slide_num: int, content: str):
    """Add content to a specific slide"""
    print(f"[PRODUCTION] Adding content to slide {slide_num}")
    return {
        "slide_id": f"slide_{slide_num}",
        "status": "updated",
        "content_preview": content[:50] + "..."
    }


# Step 2: Create agents with smart AGENT_TOOL decorator
# No need for @simulation_agent anymore - just use AGENT_TOOL!
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="slide_creator_of_smart_agent_workflow_example",
    delay_ms=500
)
def slide_creator_agent(topic: str, num_slides: int = 5):
    """Agent that creates presentation slides on a topic"""
    print(f"\n=== Slide Creator Agent ===")
    print(f"Creating {num_slides}-slide presentation on: {topic}")
    
    # Create the presentation
    presentation = create_google_slides(
        title=f"Presentation: {topic}",
        num_slides=num_slides
    )
    
    # Add content to each slide
    for i in range(1, num_slides + 1):
        slide_content = f"Slide {i}: Key points about {topic}"
        add_slide_content(
            presentation_id=presentation["presentationId"],
            slide_num=i,
            content=slide_content
        )
    
    # Return result matching production format
    return {
        "status": "success",
        "presentation_id": presentation["presentationId"],
        "url": presentation["url"],
        "slides_created": num_slides,
        "message": f"Created {num_slides}-slide presentation on {topic}"
    }


@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="research_agent_of_smart_agent_workflow_example",
    delay_ms=300
)
def research_agent(topic: str):
    """Agent that researches a topic"""
    print(f"\n=== Research Agent ===")
    print(f"Researching topic: {topic}")
    
    # Simulate research work
    time.sleep(0.1)
    
    return {
        "status": "success",
        "topic": topic,
        "key_findings": [
            f"Finding 1 about {topic}",
            f"Finding 2 about {topic}",
            f"Finding 3 about {topic}"
        ],
        "summary": f"Comprehensive research on {topic} completed"
    }


# Step 3: Create higher-level agent that coordinates others
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="workshop_organizer_of_smart_agent_workflow_example",
    delay_ms=1000
)
def workshop_organizer_agent(workshop_topic: str, audience: str):
    """Agent that organizes a complete workshop"""
    print(f"\n=== Workshop Organizer Agent ===")
    print(f"Organizing workshop on '{workshop_topic}' for {audience}")
    
    # First, research the topic
    research_result = research_agent(workshop_topic)
    
    # Then create presentation based on research
    presentation_result = slide_creator_agent(
        topic=f"{workshop_topic} for {audience}",
        num_slides=10
    )
    
    # Compile workshop plan
    workshop_plan = {
        "topic": workshop_topic,
        "audience": audience,
        "research": research_result["key_findings"],
        "presentation_id": presentation_result["presentation_id"],
        "presentation_url": presentation_result["url"],
        "agenda": [
            "Welcome and Introduction (10 min)",
            f"Overview of {workshop_topic} (20 min)",
            "Deep Dive and Examples (30 min)",
            "Interactive Exercise (20 min)",
            "Q&A Session (15 min)",
            "Wrap-up and Next Steps (5 min)"
        ]
    }
    
    return {
        "status": "success",
        "workshop_plan": workshop_plan,
        "message": f"Successfully organized workshop on {workshop_topic}"
    }


# Step 4: No need for separate tool wrappers anymore!
# The AGENT_TOOL decorator handles everything automatically.
# Just use the agent functions directly - they work as tools!


# Workflow example
def run_complete_workflow():
    """Example of tracking a complete workflow"""
    context = get_current_context()
    
    # Start the workflow
    workflow_id = "workshop_creation_001"
    workflow_name = "Complete Workshop Creation"
    context.start_workflow(workflow_id, workflow_name)
    
    print(f"\n{'='*50}")
    print(f"Starting Workflow: {workflow_name}")
    print(f"{'='*50}")
    
    try:
        result = research_agent("Machine Learning")
        print(f"\nFirst call result: {result['summary']}")
        
        # Execute the main agent
        result = workshop_organizer_agent(
            workshop_topic="Introduction to Machine Learning",
            audience="Business Leaders"
        )
        
        # End workflow successfully
        workflow_metrics = context.end_workflow()
        
        if workflow_metrics:
            print(f"\n{'='*50}")
            print("Workflow Completed Successfully!")
            print(f"{'='*50}")
            print(f"Workflow ID: {workflow_metrics.workflow_id}")
            print(f"Total Tasks: {len(workflow_metrics.tasks)}")
            print(f"Total Duration: {workflow_metrics.total_duration}ms")
            print(f"Total Tokens: {workflow_metrics.total_tokens}")
            print(f"Estimated Cost: ${workflow_metrics.total_cost:.4f}")
            print(f"Quality Score: {workflow_metrics.comment_score['score']}/10")
            
            # Save workflow metrics
            saved_path = save_workflow_metrics(workflow_metrics)
            print(f"\nMetrics saved to: {saved_path}")
            
            # Display task breakdown
            print(f"\nTask Breakdown:")
            for task in workflow_metrics.tasks:
                print(f"  - {task.task_name}: {task.total_duration}ms, {len(task.tool_calls)} tool calls")
        
        return result
        
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        context.end_workflow()
        raise


def demonstrate_smart_agent_tools():
    """Show how smart AGENT_TOOL pattern works"""
    print(f"\n\n{'='*50}")
    print("Smart Agent Tool Pattern (NEW!)")
    print(f"{'='*50}\n")
    
    print("First call to agent - will execute and save performance:")
    
    # First call - executes the agent
    result1 = research_agent("Machine Learning")
    print(f"\nFirst call result: {result1['summary']}")
    
    # Merge performance files
    merge_performance_files()
    
    print("\n\nSecond call to same agent - uses cached performance:")
    
    # Second call - uses cached performance
    result2 = research_agent("Machine Learning")
    print(f"\nSecond call result: {result2['summary']}")
    
    print("\nNote: First call executed, second call used cached data!")
    print("No need for separate @simulation_agent decorator!")


if __name__ == "__main__":
    # Set up environment
    os.environ["SIMULATION_MODE"] = "true"
    os.environ["SIMULATION_ERROR_RATE"] = "0.0"
    
    print("=== Agent and Workflow Example ===")
    
    # Demonstrate smart agent tool pattern
    demonstrate_smart_agent_tools()
    
    print("\n\nExample completed!")    
    
    # Run the complete workflow
    print("\n\nRunning complete workflow...")
    workflow_result = run_complete_workflow()
    
    print(f"\n\nFinal Workshop Plan:")
    print(f"Topic: {workflow_result['workshop_plan']['topic']}")
    print(f"Audience: {workflow_result['workshop_plan']['audience']}")
    print(f"Presentation: {workflow_result['workshop_plan']['presentation_url']}")
    
