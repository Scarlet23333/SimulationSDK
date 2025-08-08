"""
Simple metric aggregation functions for simulations.

Provides stateless functions to calculate and aggregate metrics
from TaskMetrics and WorkflowMetrics data.
"""

from typing import List, Dict, Any

from .models import TaskMetrics, ToolMetrics


def calculate_total_tokens(task_metrics: TaskMetrics) -> int:
    """
    Calculate total tokens used in a task.
    
    Args:
        task_metrics: TaskMetrics instance containing token counts
        
    Returns:
        Total tokens (LLM + tool tokens)
    """
    tool_tokens = sum(tool.tokens for tool in task_metrics.tool_calls)
    return task_metrics.llm_tokens + tool_tokens


def calculate_total_duration(task_metrics: TaskMetrics) -> int:
    """
    Calculate total duration of a task.
    
    Args:
        task_metrics: TaskMetrics instance containing durations
        
    Returns:
        Total duration in milliseconds
    """
    return task_metrics.total_duration


def aggregate_workflow_metrics(tasks: List[TaskMetrics]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple tasks in a workflow.
    
    Args:
        tasks: List of TaskMetrics from workflow execution
        
    Returns:
        Dictionary containing aggregated metrics:
        - total_tokens: Sum of all tokens used
        - total_duration: Sum of all task durations
        - task_count: Number of tasks
        - success_count: Number of successful tasks
        - average_score: Average comment score
        - tool_usage: Dictionary of tool usage counts
    """
    if not tasks:
        return {
            "total_tokens": 0,
            "total_duration": 0,
            "task_count": 0,
            "success_count": 0,
            "average_score": 0.0,
            "tool_usage": {}
        }
    
    total_tokens = sum(calculate_total_tokens(task) for task in tasks)
    total_duration = sum(task.total_duration for task in tasks)
    success_count = sum(1 for task in tasks if task.task_success)
    # Extract scores from comment_score dicts
    scores = []
    for task in tasks:
        if isinstance(task.comment_score, dict):
            scores.append(task.comment_score.get('score', 0.0))
        else:
            scores.append(task.comment_score)
    average_score = sum(scores) / len(scores) if scores else 0.0
    
    # Aggregate tool usage
    tool_usage = {}
    for task in tasks:
        for tool_call in task.tool_calls:
            tool_name = tool_call.tool_name
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_duration": 0
                }
            tool_usage[tool_name]["count"] += 1
            tool_usage[tool_name]["total_tokens"] += tool_call.tokens
            tool_usage[tool_name]["total_duration"] += tool_call.duration
    
    return {
        "total_tokens": total_tokens,
        "total_duration": total_duration,
        "task_count": len(tasks),
        "success_count": success_count,
        "average_score": average_score,
        "tool_usage": tool_usage
    }