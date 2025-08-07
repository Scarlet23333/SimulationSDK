"""
Decorators for simulation tools and agents.

This module provides the core decorators used to define simulation tools
and agents according to the simulation protocol specification.
"""

import os
import time
import random
import functools
import asyncio
import inspect
import json
from typing import Callable, Any, Optional, Tuple, Dict
from datetime import datetime
import uuid
from pathlib import Path

from .models import (
    ToolCategory,
    ToolMetrics,
    TaskMetrics,
    AgentPerformance,
    AgentSimulatedResult,
    AgentAsToolSummary,
    SimulationResponse,
)
from .registry import ToolRegistry, ResponseTemplateManager
from .context import SimulationContext
from .performance import PerformanceManager


def simulation_tool(
    category: ToolCategory,
    delay_ms: Optional[int] = None,
    delay_range: Optional[Tuple[int, int]] = None,
    success_template: Optional[SimulationResponse] = None,
    failure_template: Optional[SimulationResponse] = None,
    agent_name: Optional[str] = None,
    description: str = "",
) -> Callable:
    """
    Decorator to register a tool for simulation.
    
    When SIMULATION_MODE=true, intercepts calls based on category:
    - PRODUCTION_AFFECTING: Never executes original function, uses templates
    - READ_ONLY: Always executes original function, records real metrics
    - AGENT_TOOL: Loads performance from latest_agent_performance.json
    
    Args:
        category: Tool category (READ_ONLY, PRODUCTION_AFFECTING, AGENT_TOOL)
        delay_ms: Fixed delay in milliseconds
        delay_range: Tuple of (min, max) for random delay in milliseconds
        success_template: SimulationResponse template for successful responses (required for PRODUCTION_AFFECTING)
                        Always returns response_data field
        failure_template: SimulationResponse template for failure responses (required for PRODUCTION_AFFECTING)
                        Always raises RuntimeError with error_message field
        agent_name: Name to look up in performance data (required for AGENT_TOOL)
        description: Tool description
    
    Returns:
        Decorated function that can be simulated
    """
    # Validate required parameters
    if category == ToolCategory.PRODUCTION_AFFECTING:
        if not success_template or not failure_template:
            raise ValueError("PRODUCTION_AFFECTING tools require both success_template and failure_template")
    
    if category == ToolCategory.AGENT_TOOL:
        if not agent_name:
            raise ValueError("AGENT_TOOL category requires agent_name parameter")
    
    def decorator(func: Callable) -> Callable:
        tool_name = func.__name__
        
        # Register the tool with templates
        registry = ToolRegistry()
        registry.register(
            name=tool_name,
            category=category,
            success_template=success_template,
            failure_template=failure_template,
            description=description or func.__doc__ or "",
        )
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check if we're in simulation mode
            simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() == "true"
            
            # For read-only tools, always execute normally
            if category == ToolCategory.READ_ONLY:
                simulation_mode = False
            
            # Apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            start_time = time.time()
            
            try:
                if simulation_mode and category == ToolCategory.AGENT_TOOL:
                    # Load performance from latest_agent_performance.json
                    performance_manager = PerformanceManager()
                    agent_summary = performance_manager.load_agent_performance(agent_name)
                    
                    if agent_summary and agent_summary.simulated_result:
                        result = agent_summary.simulated_result.final_output
                        duration_ms = agent_summary.performance.duration
                    else:
                        # Fallback if no saved performance
                        result = f"Simulated {agent_name} execution"
                        duration_ms = 100
                
                elif simulation_mode and category == ToolCategory.PRODUCTION_AFFECTING:
                    # Use simulated response from templates
                    result = _simulate_production_tool(
                        func, tool_name, success_template, failure_template, args, kwargs
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                
                else:
                    # Execute real function
                    result = await func(*args, **kwargs)
                    duration_ms = int((time.time() - start_time) * 1000)
                
                # Record metrics
                _record_tool_metrics(tool_name, duration_ms, success=True)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration_ms = int((time.time() - start_time) * 1000)
                _record_tool_metrics(tool_name, duration_ms, success=False)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if we're in simulation mode
            simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() == "true"
            
            # For read-only tools, always execute normally
            if category == ToolCategory.READ_ONLY:
                simulation_mode = False
            
            # Apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                time.sleep(delay)
            
            start_time = time.time()
            
            try:
                if simulation_mode and category == ToolCategory.AGENT_TOOL:
                    # Load performance from latest_agent_performance.json
                    performance_manager = PerformanceManager()
                    agent_summary = performance_manager.load_agent_performance(agent_name)
                    
                    if agent_summary and agent_summary.simulated_result:
                        result = agent_summary.simulated_result.final_output
                        duration_ms = agent_summary.performance.duration
                    else:
                        # Fallback if no saved performance
                        result = f"Simulated {agent_name} execution"
                        duration_ms = 100
                
                elif simulation_mode and category == ToolCategory.PRODUCTION_AFFECTING:
                    # Use simulated response from templates
                    result = _simulate_production_tool(
                        func, tool_name, success_template, failure_template, args, kwargs
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                
                else:
                    # Execute real function
                    result = func(*args, **kwargs)
                    duration_ms = int((time.time() - start_time) * 1000)
                
                # Record metrics
                _record_tool_metrics(tool_name, duration_ms, success=True)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration_ms = int((time.time() - start_time) * 1000)
                _record_tool_metrics(tool_name, duration_ms, success=False)
                raise
        
        if is_async:
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Add metadata to the wrapper
        wrapper._tool_name = tool_name
        wrapper._is_simulation_tool = True
        wrapper._tool_category = category
        
        return wrapper
    
    return decorator


def _simulate_production_tool(
    func: Callable,
    tool_name: str,
    success_template: Optional[SimulationResponse],
    failure_template: Optional[SimulationResponse],
    args: tuple,
    kwargs: dict
) -> Any:
    """Simulate a production-affecting tool using templates."""
    template_manager = ResponseTemplateManager()
    
    # Extract variables from function arguments
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    variables = {}
    for i, (param_name, arg_value) in enumerate(zip(params, args)):
        variables[param_name] = arg_value
    variables.update(kwargs)
    
    # Decide whether to inject error
    if template_manager.should_inject_error() and failure_template:
        # Apply template to get the simulated response
        simulated_response = template_manager.apply_template(
            failure_template,
            variables,
            tool_name
        )
        # Raise RuntimeError with the error message
        raise RuntimeError(simulated_response.error_message or "Simulated error")
    else:
        if success_template:
            # Apply template to get the simulated response
            simulated_response = template_manager.apply_template(
                success_template,
                variables,
                tool_name
            )
            # Always return the response_data part
            return simulated_response.response_data
        else:
            # No template, return mock data directly
            return {"message": f"Simulated {tool_name} execution"}


def _record_tool_metrics(tool_name: str, duration_ms: int, success: bool) -> None:
    """Record tool execution metrics."""
    tool_metrics = ToolMetrics(
        tool_name=tool_name,
        tokens=0,  # Would need to calculate based on result size
        duration=duration_ms,
        comment_score=10 if success else 0
    )
    
    # Add metrics to context if available
    context = SimulationContext.get_current()
    if context:
        context.add_tool_metrics(tool_metrics)


def simulation_agent(
    name: Optional[str] = None,
    delay_ms: Optional[int] = None,
    delay_range: Optional[Tuple[int, int]] = None,
) -> Callable:
    """
    TEMPORARY decorator to mark a function as an agent that can be called as a tool.
    
    This decorator should only be used during testing phase and removed after.
    It creates a new SimulationContext for agent execution and saves:
    - TaskMetrics to history
    - AgentAsToolSummary to temp_performance
    
    Args:
        name: Agent name for saving (defaults to function name)
        delay_ms: Fixed delay when agent is called as tool
        delay_range: Random delay range when agent is called as tool
    
    Returns:
        Decorated agent function
    
    Example:
        @simulation_agent(name="email_writer")
        def email_writer_agent(task_goal: str, context: Dict[str, Any]) -> str:
            # Agent implementation
            return "Email content"
    """
    def decorator(func: Callable) -> Callable:
        agent_name = name or func.__name__
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Execute agent with new context
            return await _execute_agent_async(func, agent_name, args, kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                time.sleep(delay)
            
            # Execute agent with new context
            return _execute_agent_sync(func, agent_name, args, kwargs)
        
        if is_async:
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # Mark as both a simulation tool and an agent
        wrapper._agent_name = agent_name
        wrapper._is_simulation_agent = True
        
        # Also register as an AGENT_TOOL
        registry = ToolRegistry()
        registry.register(
            name=agent_name,
            category=ToolCategory.AGENT_TOOL,
            description=func.__doc__ or f"Agent: {agent_name}",
        )
        
        return wrapper
    
    return decorator


def _execute_agent_sync(func: Callable, agent_name: str, args: tuple, kwargs: dict) -> Any:
    """Execute a sync agent with performance tracking."""
    # Create new SimulationContext for agent execution
    agent_context = SimulationContext()
    agent_context.agent_name = agent_name
    
    # Start performance tracking
    start_time = time.time()
    
    # Generate task ID
    task_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Extract task goal from arguments
    task_goal = _extract_task_goal(args, kwargs)
    
    try:
        # Set the context as current
        original_context = SimulationContext.get_current()
        SimulationContext._current = agent_context
        
        # Execute the agent
        result = func(*args, **kwargs)
        
        # Save performance data
        _save_agent_performance(
            agent_name, task_id, task_goal, start_time, 
            agent_context, result, success=True
        )
        
        return result
        
    except Exception as e:
        # Save performance data even on failure
        _save_agent_performance(
            agent_name, task_id, task_goal, start_time,
            agent_context, {"error": str(e)}, success=False
        )
        raise
        
    finally:
        # Restore original context
        SimulationContext._current = original_context


async def _execute_agent_async(func: Callable, agent_name: str, args: tuple, kwargs: dict) -> Any:
    """Execute an async agent with performance tracking."""
    # Create new SimulationContext for agent execution
    agent_context = SimulationContext()
    agent_context.agent_name = agent_name
    
    # Start performance tracking
    start_time = time.time()
    
    # Generate task ID
    task_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Extract task goal from arguments
    task_goal = _extract_task_goal(args, kwargs)
    
    try:
        # Set the context as current
        original_context = SimulationContext.get_current()
        SimulationContext._current = agent_context
        
        # Execute the agent
        result = await func(*args, **kwargs)
        
        # Save performance data
        _save_agent_performance(
            agent_name, task_id, task_goal, start_time,
            agent_context, result, success=True
        )
        
        return result
        
    except Exception as e:
        # Save performance data even on failure
        _save_agent_performance(
            agent_name, task_id, task_goal, start_time,
            agent_context, {"error": str(e)}, success=False
        )
        raise
        
    finally:
        # Restore original context
        SimulationContext._current = original_context


def _extract_task_goal(args: tuple, kwargs: dict) -> str:
    """Extract task goal from function arguments."""
    if args:
        return str(args[0])
    elif "task_goal" in kwargs:
        return kwargs["task_goal"]
    elif "task" in kwargs:
        return str(kwargs["task"])
    else:
        return "Unknown task"


def _save_agent_performance(
    agent_name: str,
    task_id: str,
    task_goal: str,
    start_time: float,
    agent_context: SimulationContext,
    result: Any,
    success: bool
) -> None:
    """Save agent performance data."""
    # Calculate performance metrics
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Get tool metrics from context
    tool_metrics = []
    if hasattr(agent_context, '_tool_metrics'):
        tool_metrics = agent_context._tool_metrics
    
    # Calculate total tokens
    tool_tokens = sum(tm.tokens for tm in tool_metrics)
    llm_tokens = 100 if success else 50  # Default estimate
    total_tokens = llm_tokens + tool_tokens
    
    # Create task metrics
    task_metrics = TaskMetrics(
        task_id=task_id,
        task_name=agent_name,
        task_success=success,
        llm_tokens=llm_tokens,
        tool_calls=tool_metrics,
        total_tokens=total_tokens,
        total_duration=duration_ms,
        comment_score=10 if success else 0
    )
    
    # Save task metrics to history
    if hasattr(agent_context, 'save_task_metrics'):
        agent_context.save_task_metrics(task_metrics)
    
    # Create agent performance summary
    agent_performance = AgentPerformance(
        tokens=total_tokens,
        duration=duration_ms,
        comment_score=10 if success else 0,
    )
    
    # Create simulated result
    simulated_result = AgentSimulatedResult(
        task_goal=task_goal,
        langraph_execution={},  # Empty for now
        tool_calls=[tm.model_dump() for tm in tool_metrics],
        final_output=result,
    )
    
    # Create agent as tool summary
    agent_summary = AgentAsToolSummary(
        tool_name=agent_name,
        performance=agent_performance,
        simulated_result=simulated_result,
    )
    
    # Save performance using PerformanceManager
    performance_manager = PerformanceManager()
    performance_manager.save_temp_performance(agent_name, agent_summary)