"""
Simulation context for tracking agent and tool execution.

This module provides thread-local storage for the current simulation state
and tracks metrics for the current agent/task execution.
"""

import threading
from typing import List, Optional
from datetime import datetime

from .models import ToolMetrics, TaskMetrics
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