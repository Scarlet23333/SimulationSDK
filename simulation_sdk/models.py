"""
Core data models for the SimulationSDK.

This module defines the Pydantic models used throughout the SDK
for type safety and data validation according to the simulation protocol.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict
from enum import Enum


class ToolCategory(str, Enum):
    """Categories of tools in the simulation system."""
    READ_ONLY = "read_only"
    PRODUCTION_AFFECTING = "production_affecting"
    AGENT_TOOL = "agent_tool"


class ToolMetrics(BaseModel):
    """Metrics for a single tool call."""
    tool_name: str
    tokens: int
    duration: int  # milliseconds
    comment_score: float = 10.0  # default 10.0, range 0.0-10.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskMetrics(BaseModel):
    """Metrics for a complete task execution."""
    task_id: str
    task_name: str
    task_success: bool
    llm_tokens: int
    tool_calls: List[ToolMetrics]
    total_tokens: int  # llm_tokens + sum of tool tokens
    total_duration: int  # milliseconds
    comment_score: float  # 0.0-10.0, from LLM evaluator
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowMetrics(BaseModel):
    """Metrics for an entire workflow execution."""
    workflow_id: str
    workflow_name: str
    workflow_success: bool
    tasks: List[TaskMetrics]
    total_tokens: int  # Sum of all task tokens
    total_duration: int  # milliseconds
    total_cost: float  # total_tokens * price_per_token
    comment_score: float  # 0.0-10.0, from LLM evaluator
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentPerformance(BaseModel):
    """Performance metrics for an agent execution."""
    tokens: int
    duration: int  # milliseconds
    comment_score: float
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentSimulatedResult(BaseModel):
    """Complete simulated result for an agent execution."""
    tool_calls: List[Dict[str, Any]]  # Tool call history
    final_output: Any  # Must match real agent's response format
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentAsToolSummary(BaseModel):
    """Summary of agent performance when used as a tool by another agent."""
    tool_name: str
    performance: AgentPerformance
    simulated_result: AgentSimulatedResult
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimulationResponse(BaseModel):
    """Response template for simulated tool calls."""
    success: bool
    response_data: Dict[str, Any]
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)