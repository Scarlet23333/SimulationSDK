"""
Example demonstrating automatic evaluation in SimulationSDK.

This example shows how the SDK automatically evaluates agent and workflow performance
using either OpenAI's GPT models (if OPENAI_API_KEY is set) or a MockEvaluator.
"""

import os
from dotenv import load_dotenv
from simulation_sdk import simulation_tool, simulation_agent, ToolCategory, get_current_context, save_workflow_metrics, merge_performance_files

# Load environment variables from .env file
load_dotenv()

# Enable simulation mode
os.environ["SIMULATION_MODE"] = "true"


@simulation_tool(
    category=ToolCategory.READ_ONLY,
    description="Search for documents"
)
def search_documents(query: str) -> list:
    """Search for documents matching the query."""
    return [
        {"id": 1, "title": "Document 1", "relevance": 0.95},
        {"id": 2, "title": "Document 2", "relevance": 0.87}
    ]


@simulation_tool(
    category=ToolCategory.READ_ONLY,
    description="Extract key information from document"
)
def extract_info(doc_id: int) -> dict:
    """Extract information from a document."""
    return {
        "doc_id": doc_id,
        "summary": "Important findings from the document",
        "key_points": ["Point 1", "Point 2", "Point 3"]
    }


@simulation_agent(
    name="research_assistant_of_automatic_evaluation_example",
    delay_ms=1000
)
def research_topic(topic: str) -> dict:
    """Research a topic and provide summary."""
    # Search for relevant documents
    docs = search_documents(f"query: {topic}")
    
    # Extract information from top documents
    summaries = []
    for doc in docs[:2]:
        info = extract_info(doc["id"])
        summaries.append(info)
    
    # Compile research results
    return {
        "topic": topic,
        "documents_reviewed": len(summaries),
        "key_findings": [s["summary"] for s in summaries],
        "confidence": 0.85
    }


def main():
    """Demonstrate automatic evaluation."""
    print("=== SimulationSDK Automatic Evaluation Example ===\n")
    
    # Check evaluation mode    
    if os.environ.get("OPENAI_API_KEY"):
        print("✓ OpenAI API key detected - will use GPT for evaluation")
    else:
        print("ℹ No OpenAI API key - will use MockEvaluator")
        print("  (Set OPENAI_API_KEY environment variable for LLM evaluation)\n")
    
    # Workflow execution with automatic evaluation
    print("\nExecuting workflow with multiple tasks:")
    context = get_current_context()
    context.start_workflow("research_workflow_001", "Multi-Topic Research")
    
    # Execute multiple research tasks
    topics = ["machine learning", "quantum computing", "biotechnology"]
    for topic in topics:
        result = research_topic(topic)
        print(f"   ✓ Researched: {topic}")
    
    # End workflow - automatic evaluation will occur
    workflow_metrics = context.end_workflow(success=True)
    
    merge_performance_files()
    
    if workflow_metrics:
        print(f"\n   Workflow automatically evaluated:")
        print(f"   ✓ Overall score: {workflow_metrics.comment_score}/10")
        print(f"   ✓ Total tokens: {workflow_metrics.total_tokens}")
        print(f"   ✓ Estimated cost: ${workflow_metrics.total_cost:.4f}")
        
        # Show individual task scores
        print(f"\n   Individual task scores:")
        for i, (topic, task) in enumerate(zip(topics, workflow_metrics.tasks)):
            print(f"   - {topic}: {task.comment_score}/10")
        save_workflow_metrics(workflow_metrics)
    
    print("\n✓ Example completed!")
    print("\nNote: Scores are automatically assigned by the evaluator.")
    print("With OPENAI_API_KEY set, you get intelligent LLM-based evaluation.")
    print("Without it, MockEvaluator provides reasonable placeholder scores.")


if __name__ == "__main__":
    main()
    
    # Cleanup
    if "SIMULATION_MODE" in os.environ:
        del os.environ["SIMULATION_MODE"]