"""
Simulation context for tracking agent and tool execution.

This module provides thread-local storage for the current simulation state
and tracks metrics for the current agent/task execution.
"""

import threading
from typing import List, Optional
from datetime import datetime
import uuid

from .models import ToolMetrics, TaskMetrics, WorkflowMetrics
from .utils import logger


# Thread-local storage for current simulation context
_thread_local = threading.local()


class SimulationContext:
    """
    Thread-local context for tracking current simulation state.
    
    Tracks metrics for the current agent/task execution and provides
    simple stack-like behavior for nested agents.
    """
    
    def __init__(self):
        """
        Initialize simulation context.
        """
        self.agent_name: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.tool_metrics: List[ToolMetrics] = []
        self.llm_tokens: int = 0
        
        # Stack for nested agent tracking
        self.agent_stack: List[str] = []
        
        # Workflow tracking
        self.workflow_id: Optional[str] = None
        self.workflow_name: Optional[str] = None
        self.workflow_start_time: Optional[datetime] = None
        self.workflow_tasks: List[TaskMetrics] = []
        
    def start_agent(self, agent_name: str) -> None:
        """
        Start tracking an agent execution.
        
        Called by @simulation_agent decorator.
        
        Args:
            agent_name: Name of the agent being executed
        """
        self.agent_name = agent_name
        self.start_time = datetime.now()
        self.agent_stack.append(agent_name)
        logger.debug(f"Started agent: {agent_name}")
        
    def add_tool_metrics(self, metrics: ToolMetrics) -> None:
        """
        Add tool metrics to the current context.
        
        Called by @simulation_tool decorator.
        
        Args:
            metrics: ToolMetrics instance to add
        """
        self.tool_metrics.append(metrics)
        logger.debug(f"Added tool metrics: {metrics.tool_name}")
        
    def add_llm_tokens(self, tokens: int) -> None:
        """
        Add LLM token usage to the current context.
        
        Args:
            tokens: Number of tokens used
        """
        self.llm_tokens += tokens
        logger.debug(f"Added {tokens} LLM tokens (total: {self.llm_tokens})")
        
    def get_task_metrics(self) -> TaskMetrics:
        """
        Get the current task metrics.
        
        Returns:
            TaskMetrics for the current agent execution
        """
        if not self.agent_name or not self.start_time:
            raise ValueError("No agent execution in progress")
            
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return TaskMetrics(
            agent_name=self.agent_name,
            goal="",  # Will be set by the agent
            duration_seconds=duration,
            llm_tokens_used=self.llm_tokens,
            tool_calls=self.tool_metrics,
            success=True,  # Will be updated on failure
            error=None,
            timestamp=self.start_time,
        )
        
    def start_workflow(self, workflow_id: str, workflow_name: str) -> None:
        """
        Start tracking a workflow execution.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_name: Human-readable workflow name
        """
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.workflow_start_time = datetime.now()
        self.workflow_tasks = []
        logger.debug(f"Started workflow: {workflow_name} (ID: {workflow_id})")
        
    def add_task_metrics(self, task_metrics: TaskMetrics) -> None:
        """
        Add task metrics to the current workflow.
        
        Args:
            task_metrics: TaskMetrics instance to add to workflow
        """
        if self.workflow_id:
            self.workflow_tasks.append(task_metrics)
            logger.debug(f"Added task metrics to workflow: {task_metrics.task_name}")
            
    def end_workflow(self, success: bool, comment_score: int = 10) -> Optional[WorkflowMetrics]:
        """
        End the current workflow and return WorkflowMetrics.
        
        Args:
            success: Whether the workflow succeeded overall
            comment_score: LLM evaluation score (0-10)
            
        Returns:
            WorkflowMetrics for the completed workflow, or None if no workflow
        """
        if not self.workflow_id or not self.workflow_start_time:
            logger.warning("No workflow in progress")
            return None
            
        # Calculate totals
        total_tokens = sum(task.total_tokens for task in self.workflow_tasks)
        duration_ms = int((datetime.now() - self.workflow_start_time).total_seconds() * 1000)
        
        # Simple cost calculation (example: $0.002 per 1K tokens)
        total_cost = (total_tokens / 1000) * 0.002
        
        workflow_metrics = WorkflowMetrics(
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            workflow_success=success,
            tasks=self.workflow_tasks,
            total_tokens=total_tokens,
            total_duration=duration_ms,
            total_cost=total_cost,
            comment_score=comment_score
        )
        
        # Clear workflow state
        self.workflow_id = None
        self.workflow_name = None
        self.workflow_start_time = None
        self.workflow_tasks = []
        
        logger.debug(f"Ended workflow: {workflow_metrics.workflow_name}")
        return workflow_metrics
        
    def clear(self) -> None:
        """
        Clear the context after agent completes.
        
        Resets all tracking state.
        """
        if self.agent_stack and self.agent_name:
            self.agent_stack.pop()
            
        self.agent_name = None
        self.start_time = None
        self.tool_metrics = []
        self.llm_tokens = 0
        logger.debug("Cleared simulation context")
        
    @classmethod
    def get_current(cls) -> Optional['SimulationContext']:
        """
        Get the current simulation context.
        
        Auto-creates a context if none exists.
        
        Returns:
            Current SimulationContext or newly created one
        """
        if not hasattr(_thread_local, 'context') or _thread_local.context is None:
            _thread_local.context = cls()
            
        return _thread_local.context
        
    @classmethod
    def clear_current(cls) -> None:
        """
        Clear the current context from thread-local storage.
        """
        if hasattr(_thread_local, 'context'):
            _thread_local.context = None


# Convenience function for getting current context
def get_current_context() -> Optional[SimulationContext]:
    """
    Get the current simulation context.
    
    This is a convenience function that calls SimulationContext.get_current().
    
    Returns:
        Current SimulationContext or newly created one
    """
    return SimulationContext.get_current()