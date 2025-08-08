"""
LLM Token Tracking Example - SimulationSDK

This example shows how to automatically track LLM tokens in your agents.
"""

import os
from simulation_sdk import (
    simulation_tool,
    ToolCategory,
    SimulationResponse,
    get_current_context,
    track_llm_tokens,
    merge_performance_files
)

# Enable simulation mode
os.environ["SIMULATION_MODE"] = "true"


# Example 1: Simple LangGraph Tools
@simulation_tool(
    category=ToolCategory.READ_ONLY,
    delay_ms=100
)
def search_knowledge_base(query: str) -> dict:
    """Search internal knowledge base"""
    print(f"[TOOL] Searching for: {query}")
    return {
        "results": [
            {"title": "Python Guide", "relevance": 0.9},
            {"title": "LangGraph Tutorial", "relevance": 0.8}
        ]
    }


@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,
    success_template=SimulationResponse(
        success=True,
        response_data={
            "doc_id": "doc_123",
            "title": "$title",
            "status": "created"
        }
    ),
    failure_template=SimulationResponse(
        success=False,
        response_data={},
        error_message="Failed to create document"
    ),
    delay_ms=300
)
def create_document(title: str, content: str) -> dict:
    """Create a new document"""
    print(f"[PRODUCTION] Creating document: {title}")
    return {"doc_id": "real_doc_456", "status": "created"}


# Example 2: LLM call with automatic token tracking
@track_llm_tokens
def call_llm(prompt: str) -> dict:
    """
    Simulates an LLM call that returns a response with token usage.
    In real usage, this would be openai.chat.completions.create() or similar.
    """
    print(f"[LLM] Processing: {prompt}")
    
    # Simulate LLM response with token usage
    return {
        "content": f"Analysis of '{prompt}': This topic is highly relevant.",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150
        }
    }


# Example 3: Smart Agent Tool with automatic token tracking
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="research_agent_of_token_tracking_example",
    delay_ms=1000
)
def research_agent(topic: str) -> dict:
    """
    Research agent that uses LLM and tools.
    First call will execute and save performance.
    Later calls use cached performance.
    """
    print(f"\n=== Research Agent ===")
    print(f"Researching topic: {topic}")
    
    # Search for relevant information
    search_results = search_knowledge_base(topic)
    
    # Make LLM call - tokens tracked automatically!
    llm_response = call_llm(f"Analyze this topic: {topic}")
    
    return {
        "topic": topic,
        "analysis": llm_response["content"],
        "sources": search_results["results"],
        "tokens_used": llm_response["usage"]["total_tokens"]
    }


# Example 4: Content Creator Agent
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="content_creator_of_token_tracking_example",
    delay_ms=1500
)
def content_creator_agent(research: dict) -> dict:
    """
    Creates content based on research.
    Demonstrates agent-to-agent coordination.
    """
    print(f"\n=== Content Creator Agent ===")
    print(f"Creating content from research on: {research['topic']}")
    
    # Generate content using LLM - tokens tracked automatically!
    content_response = call_llm(f"Write content about: {research['analysis']}")
    
    # Create document
    doc_result = create_document(
        title=f"Guide to {research['topic']}",
        content=content_response["content"]
    )
    
    return {
        "document": doc_result,
        "content": content_response["content"],
        "status": "published",
        "tokens_used": content_response["usage"]["total_tokens"]
    }


# Example 5: Workflow with Automatic Token Tracking
def simulate_workflow_with_tracking():
    """
    Simulates a workflow with automatic token tracking.
    Shows how tokens are accumulated across agent calls.
    """
    print("=== Workflow with Token Tracking Example ===\n")
    
    # Simulated workflow execution
    context = get_current_context()
    context.start_workflow("langgraph_wf_001", "Content Generation Workflow")
    
    try:
        # Step 1: Research
        print("Step 1: Research Phase")
        research_result = research_agent("AI Safety")
        
        # Step 2: Content Creation
        print("\nStep 2: Content Creation Phase")
        content_result = content_creator_agent(research_result)
        
        # End workflow
        metrics = context.end_workflow()
        
        if metrics:
            print(f"\n=== Workflow Metrics ===")
            print(f"Total Duration: {metrics.total_duration}ms")
            print(f"Total Tokens: {metrics.total_tokens}")
            print(f"Total Cost: ${metrics.total_cost:.4f}")
            
            # Show task breakdown
            print(f"\nTasks ({len(metrics.tasks)}):")
            for task in metrics.tasks:
                print(f"  - {task.task_name}: {task.llm_tokens} tokens, {len(task.tool_calls)} tools")
        
        return content_result
        
    except Exception as e:
        print(f"Workflow failed: {e}")
        context.end_workflow()
        raise


# Example 6: Real-world Usage Examples
def demonstrate_real_world_usage():
    """Shows how to use token tracking with real LLM libraries"""
    print("\n\n=== Real-World Usage Examples ===\n")
    
    print("1. With OpenAI:")
    print("```python")
    print("@track_llm_tokens")
    print("def call_openai(prompt: str):")
    print("    return openai.chat.completions.create(")
    print("        model='gpt-4',")
    print("        messages=[{'role': 'user', 'content': prompt}]")
    print("    )")
    print("```")
    
    print("\n2. With LangChain:")
    print("```python")
    print("@track_llm_tokens")
    print("def call_langchain(prompt: str):")
    print("    return llm_chain.invoke({'input': prompt})")
    print("```")
    
    print("\n3. With Custom LLM:")
    print("```python")
    print("@track_llm_tokens")
    print("def call_custom_llm(prompt: str):")
    print("    response = your_llm_api(prompt)")
    print("    # Ensure response has 'usage' field with 'total_tokens'")
    print("    return response")
    print("```")


if __name__ == "__main__":
    # Run workflow simulation
    workflow_result = simulate_workflow_with_tracking()
    
    print(f"\n\nFinal Result: Document created with ID {workflow_result['document']['doc_id']}")
    
    # Demonstrate real-world usage
    demonstrate_real_world_usage()
    
    # Merge performance files
    merge_performance_files()
    
    print("\n\n=== Key Takeaways ===")
    print("1. Use @track_llm_tokens decorator on any LLM call function")
    print("2. Tokens are automatically added to the parent agent's context")
    print("3. AGENT_TOOL auto-executes on first call (no manual setup)")
    print("4. All metrics are tracked hierarchically (workflow → task → tool)")
    print("5. No need to specify agent_name - context is automatic!")
    
    print("\nExample completed!")