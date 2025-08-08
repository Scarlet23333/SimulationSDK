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
from .context import SimulationContext, _thread_local
from .performance import PerformanceManager
from .evaluator import OpenAIEvaluator, MockEvaluator, EvaluatorInterface
from .utils import logger

# Global evaluator instance - created once at module level
_evaluator: Optional[EvaluatorInterface] = None

def _get_evaluator() -> EvaluatorInterface:
    """Get or create the global evaluator instance."""
    global _evaluator
    if _evaluator is None:
        # Check for OpenAI API key in environment
        if os.environ.get("OPENAI_API_KEY"):
            try:
                _evaluator = OpenAIEvaluator()
                logger.info("Using OpenAIEvaluator for automatic evaluation")
            except Exception as e:
                logger.warning(f"Failed to create OpenAIEvaluator: {e}, falling back to MockEvaluator")
                _evaluator = MockEvaluator()
        else:
            _evaluator = MockEvaluator()
            logger.info("Using MockEvaluator for automatic evaluation (set OPENAI_API_KEY for LLM evaluation)")
    return _evaluator


def track_llm_tokens(func: Callable) -> Callable:
    """
    Decorator that automatically tracks LLM token usage.
    Updates the parent simulation context (if any).
    
    Usage:
        @track_llm_tokens
        def my_llm_call(prompt: str):
            response = openai.chat.completions.create(...)
            return response
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Execute the LLM call
        response = func(*args, **kwargs)
        
        # Get current context (from parent agent/tool)
        context = SimulationContext.get_current()
        if context:  # We're inside an agent or tool
            # Extract tokens from various response formats
            tokens = 0
            
            # OpenAI format
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                tokens = response.usage.total_tokens
            # LangChain format
            elif hasattr(response, 'llm_output') and isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})
                tokens = token_usage.get('total_tokens', 0)
            # Dict format
            elif isinstance(response, dict):
                usage = response.get('usage', {})
                if isinstance(usage, dict):
                    tokens = usage.get('total_tokens', 0)
                elif hasattr(usage, 'total_tokens'):
                    tokens = usage.total_tokens
            
            # Add tokens to current context
            if tokens > 0:
                context.add_llm_tokens(tokens)
        
        return response
    
    # Support async functions
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Execute the LLM call
        response = await func(*args, **kwargs)
        
        # Get current context (from parent agent/tool)
        context = SimulationContext.get_current()
        if context:  # We're inside an agent or tool
            # Extract tokens from various response formats
            tokens = 0
            
            # OpenAI format
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                tokens = response.usage.total_tokens
            # LangChain format
            elif hasattr(response, 'llm_output') and isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get('token_usage', {})
                tokens = token_usage.get('total_tokens', 0)
            # Dict format
            elif isinstance(response, dict):
                usage = response.get('usage', {})
                if isinstance(usage, dict):
                    tokens = usage.get('total_tokens', 0)
                elif hasattr(usage, 'total_tokens'):
                    tokens = usage.total_tokens
            
            # Add tokens to current context
            if tokens > 0:
                context.add_llm_tokens(tokens)
        
        return response
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


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
            
            # Fast path for production mode - minimal overhead
            if not simulation_mode:
                if category == ToolCategory.READ_ONLY:
                    # Execute and return immediately
                    return await func(*args, **kwargs)
                elif category == ToolCategory.PRODUCTION_AFFECTING:
                    # Execute real function in production
                    return await func(*args, **kwargs)
                elif category == ToolCategory.AGENT_TOOL:
                    # Execute real function in production
                    return await func(*args, **kwargs)
            
            # For read-only tools in simulation mode, always execute but record metrics
            if category == ToolCategory.READ_ONLY:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = int((time.time() - start_time) * 1000)
                    _record_tool_metrics(tool_name, duration_ms, success=True)
                    return result
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    _record_tool_metrics(tool_name, duration_ms, success=False)
                    raise
            
            # Only in simulation mode from here - apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            start_time = time.time()
            
            try:
                if category == ToolCategory.AGENT_TOOL:
                    # Load performance from latest_agent_performance.json
                    performance_manager = PerformanceManager()
                    agent_summary = performance_manager.load_agent_performance(agent_name)
                    
                    if agent_summary and agent_summary.simulated_result:
                        result = agent_summary.simulated_result.final_output
                        duration_ms = agent_summary.performance.duration
                        # Record as both tool metrics and task metrics
                        _record_agent_tool_as_task(agent_name, agent_summary, duration_ms, True, result)
                    else:
                        # No saved performance - execute as simulation_agent first
                        # This will automatically create TaskMetrics and add to workflow
                        agent_wrapper = simulation_agent(name=agent_name)(func)
                        result = await agent_wrapper(*args, **kwargs)
                        
                        # Merge temp files to make performance available
                        performance_manager.merge_temp_to_latest()
                        
                        # Now load the saved performance for duration
                        agent_summary = performance_manager.load_agent_performance(agent_name)
                        if agent_summary:
                            duration_ms = agent_summary.performance.duration
                        else:
                            duration_ms = int((time.time() - start_time) * 1000)
                        # Don't record again - simulation_agent already did it
                
                elif category == ToolCategory.PRODUCTION_AFFECTING:
                    # Use simulated response from templates
                    result = _simulate_production_tool(
                        func, tool_name, success_template, failure_template, args, kwargs
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                
                else:
                    # Should not reach here in simulation mode
                    raise ValueError(f"Unknown tool category: {category}")
                
                # Record metrics (we're in simulation mode)
                _record_tool_metrics(tool_name, duration_ms, success=True)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration_ms = int((time.time() - start_time) * 1000)
                _record_tool_metrics(tool_name, duration_ms, success=False)
                
                # For AGENT_TOOL, also record as failed task
                if category == ToolCategory.AGENT_TOOL:
                    _record_agent_tool_as_task(agent_name, None, duration_ms, False, {"error": str(e)})
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check if we're in simulation mode
            simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() == "true"
            
            # Fast path for production mode - minimal overhead
            if not simulation_mode:
                if category == ToolCategory.READ_ONLY:
                    # Execute and return immediately
                    return func(*args, **kwargs)
                elif category == ToolCategory.PRODUCTION_AFFECTING:
                    # Execute real function in production
                    return func(*args, **kwargs)
                elif category == ToolCategory.AGENT_TOOL:
                    # Execute real function in production
                    return func(*args, **kwargs)
            
            # For read-only tools in simulation mode, always execute but record metrics
            if category == ToolCategory.READ_ONLY:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = int((time.time() - start_time) * 1000)
                    _record_tool_metrics(tool_name, duration_ms, success=True)
                    return result
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    _record_tool_metrics(tool_name, duration_ms, success=False)
                    raise
            
            # Only in simulation mode from here - apply delay if specified
            delay = 0
            if delay_ms is not None:
                delay = delay_ms / 1000.0
            elif delay_range is not None:
                delay = random.randint(delay_range[0], delay_range[1]) / 1000.0
            
            if delay > 0:
                time.sleep(delay)
            
            start_time = time.time()
            
            try:
                if category == ToolCategory.AGENT_TOOL:
                    # Load performance from latest_agent_performance.json
                    performance_manager = PerformanceManager()
                    agent_summary = performance_manager.load_agent_performance(agent_name)
                    
                    if agent_summary and agent_summary.simulated_result:
                        result = agent_summary.simulated_result.final_output
                        duration_ms = agent_summary.performance.duration
                        # Record as both tool metrics and task metrics
                        _record_agent_tool_as_task(agent_name, agent_summary, duration_ms, True, result)
                    else:
                        # No saved performance - execute as simulation_agent first
                        # This will automatically create TaskMetrics and add to workflow
                        agent_wrapper = simulation_agent(name=agent_name)(func)
                        result = agent_wrapper(*args, **kwargs)
                        
                        # Merge temp files to make performance available
                        performance_manager.merge_temp_to_latest()
                        
                        # Now load the saved performance for duration
                        agent_summary = performance_manager.load_agent_performance(agent_name)
                        if agent_summary:
                            duration_ms = agent_summary.performance.duration
                        else:
                            duration_ms = int((time.time() - start_time) * 1000)
                        # Don't record again - simulation_agent already did it
                
                elif category == ToolCategory.PRODUCTION_AFFECTING:
                    # Use simulated response from templates
                    result = _simulate_production_tool(
                        func, tool_name, success_template, failure_template, args, kwargs
                    )
                    duration_ms = int((time.time() - start_time) * 1000)
                
                else:
                    # Should not reach here in simulation mode
                    raise ValueError(f"Unknown tool category: {category}")
                
                # Record metrics (we're in simulation mode)
                _record_tool_metrics(tool_name, duration_ms, success=True)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration_ms = int((time.time() - start_time) * 1000)
                _record_tool_metrics(tool_name, duration_ms, success=False)
                
                # For AGENT_TOOL, also record as failed task
                if category == ToolCategory.AGENT_TOOL:
                    _record_agent_tool_as_task(agent_name, None, duration_ms, False, {"error": str(e)})
                
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
    # Get tool category from registry
    registry = ToolRegistry()
    category = registry.get_category(tool_name)
    
    # Simple scoring for non-agent tools
    if category != ToolCategory.AGENT_TOOL:
        comment_score = 10.0 if success else 0.0
    else:
        # AGENT_TOOL will get score from evaluator later
        comment_score = 10.0 if success else 0.0  # Default, will be updated
    
    tool_metrics = ToolMetrics(
        tool_name=tool_name,
        tokens=0,  # Would need to calculate based on result size
        duration=duration_ms,
        comment_score=comment_score
    )
    
    # Add metrics to context if available
    context = SimulationContext.get_current()
    if context:
        context.add_tool_metrics(tool_metrics)


def _record_agent_tool_as_task(
    agent_name: str, 
    agent_summary: Optional[AgentAsToolSummary], 
    duration_ms: int,
    success: bool,
    result: Any
) -> None:
    """Record AGENT_TOOL execution as a task in the workflow."""
    # Generate task ID
    task_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Extract metrics from agent summary if available
    if agent_summary and agent_summary.simulated_result:
        tool_calls = [
            ToolMetrics(**tc) if isinstance(tc, dict) else tc
            for tc in agent_summary.simulated_result.tool_calls
        ]
        total_tokens = agent_summary.performance.tokens
        llm_tokens = max(0, total_tokens - sum(tm.tokens for tm in tool_calls))
        # Use existing score if available
        existing_score = agent_summary.performance.comment_score
    else:
        # No summary available, use defaults
        tool_calls = []
        llm_tokens = 100 if success else 50
        total_tokens = llm_tokens
        existing_score = None
    
    # Evaluate task performance automatically if no existing score
    if existing_score is not None:
        score = existing_score
    else:
        try:
            evaluator = _get_evaluator()
            evaluation_context = {
                "agent_name": agent_name,
                "task_success": success,
            }
            if agent_summary:
                evaluation_context["agent_summary"] = {
                    "tool_calls": [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in tool_calls],
                    "final_output": result,
                    "performance": {
                        "tokens": total_tokens,
                        "duration": duration_ms
                    }
                }
            score, reasoning = evaluator.evaluate_task(evaluation_context)
            logger.debug(f"Task '{agent_name}' evaluated: score={score}, reason={reasoning}")
        except Exception as e:
            logger.warning(f"Failed to evaluate task '{agent_name}': {e}, using default score")
            score = 10.0 if success else 0.0
    
    # Create task metrics with evaluated score
    task_metrics = TaskMetrics(
        task_id=task_id,
        task_name=agent_name,
        task_success=success,
        llm_tokens=llm_tokens,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        total_duration=duration_ms,
        comment_score=score
    )
    
    # Add task metrics to current workflow context
    context = SimulationContext.get_current()
    if context and context.workflow_id:
        context.add_task_metrics(task_metrics)


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
    # Check if we're in simulation mode
    simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() == "true"
    
    # If not in simulation mode, just execute the function directly
    if not simulation_mode:
        return func(*args, **kwargs)
    
    # Only create context and track performance in simulation mode
    # Create new SimulationContext for agent execution
    agent_context = SimulationContext()
    agent_context.agent_name = agent_name
    
    # Start performance tracking
    start_time = time.time()
    
    # Generate task ID
    task_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Set the context as current
        original_context = SimulationContext.get_current()
        _thread_local.context = agent_context
        
        # Execute the agent
        result = func(*args, **kwargs)
        
        # Save performance data
        _save_agent_performance(
            agent_name, task_id, start_time, 
            agent_context, result, success=True, original_context=original_context
        )
        
        return result
        
    except Exception as e:
        # Save performance data even on failure
        _save_agent_performance(
            agent_name, task_id, start_time,
            agent_context, {"error": str(e)}, success=False, original_context=original_context
        )
        raise
        
    finally:
        # Restore original context
        _thread_local.context = original_context


async def _execute_agent_async(func: Callable, agent_name: str, args: tuple, kwargs: dict) -> Any:
    """Execute an async agent with performance tracking."""
    # Check if we're in simulation mode
    simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() == "true"
    
    # If not in simulation mode, just execute the function directly
    if not simulation_mode:
        return await func(*args, **kwargs)
    
    # Only create context and track performance in simulation mode
    # Create new SimulationContext for agent execution
    agent_context = SimulationContext()
    agent_context.agent_name = agent_name
    
    # Start performance tracking
    start_time = time.time()
    
    # Generate task ID
    task_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Set the context as current
        original_context = SimulationContext.get_current()
        _thread_local.context = agent_context
        
        # Execute the agent
        result = await func(*args, **kwargs)
        
        # Save performance data
        _save_agent_performance(
            agent_name, task_id, start_time,
            agent_context, result, success=True, original_context=original_context
        )
        
        return result
        
    except Exception as e:
        # Save performance data even on failure
        _save_agent_performance(
            agent_name, task_id, start_time,
            agent_context, {"error": str(e)}, success=False, original_context=original_context
        )
        raise
        
    finally:
        # Restore original context
        _thread_local.context = original_context




def _save_agent_performance(
    agent_name: str,
    task_id: str,
    start_time: float,
    agent_context: SimulationContext,
    result: Any,
    success: bool,
    original_context: Optional[SimulationContext] = None
) -> None:
    """Save agent performance data (only called when SIMULATION_MODE=true)."""
    # Calculate performance metrics
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Get tool metrics from context
    tool_metrics = agent_context.tool_metrics
    
    # Calculate total tokens
    tool_tokens = sum(tm.tokens for tm in tool_metrics)
    # Get LLM tokens from context (tracked by @track_llm_tokens)
    llm_tokens = getattr(agent_context, 'llm_tokens', 0)
    # Use default only if no tokens were tracked
    if llm_tokens == 0:
        llm_tokens = 100 if success else 50  # Default estimate
    total_tokens = llm_tokens + tool_tokens
    
    # Create simulated result for evaluation
    simulated_result = AgentSimulatedResult(
        tool_calls=[tm.model_dump() for tm in tool_metrics],
        final_output=result,
    )
    
    # Evaluate agent performance automatically
    try:
        evaluator = _get_evaluator()
        evaluation_context = {
            "agent_name": agent_name,
            "task_success": success,
            "agent_summary": {
                "tool_calls": [tm.model_dump() for tm in tool_metrics],
                "final_output": result,
                "performance": {
                    "tokens": total_tokens,
                    "duration": duration_ms
                }
            }
        }
        score, reasoning = evaluator.evaluate_task(evaluation_context)
        logger.debug(f"Agent '{agent_name}' evaluated: score={score}, reason={reasoning}")
    except Exception as e:
        logger.warning(f"Failed to evaluate agent '{agent_name}': {e}, using default score")
        score = 10.0 if success else 0.0
        reasoning = "Evaluation failed, using default score"
    
    # Create task metrics with evaluated score
    task_metrics = TaskMetrics(
        task_id=task_id,
        task_name=agent_name,
        task_success=success,
        llm_tokens=llm_tokens,
        tool_calls=tool_metrics,
        total_tokens=total_tokens,
        total_duration=duration_ms,
        comment_score=score
    )
    
    # Save task metrics to history
    if hasattr(agent_context, 'save_task_metrics'):
        agent_context.save_task_metrics(task_metrics)
    
    # Save task history using PerformanceManager
    performance_manager = PerformanceManager()
    performance_manager.save_task_history(agent_name, task_metrics)
    
    # Add task metrics to workflow if one is active
    # Use the original context which may have workflow tracking
    if original_context and original_context.workflow_id:
        original_context.add_task_metrics(task_metrics)
    
    # Create agent performance summary with evaluated score
    agent_performance = AgentPerformance(
        tokens=total_tokens,
        duration=duration_ms,
        comment_score=score,
    )
    
    # Create agent as tool summary
    agent_summary = AgentAsToolSummary(
        tool_name=agent_name,
        performance=agent_performance,
        simulated_result=simulated_result,
    )
    
    # Save performance using PerformanceManager
    performance_manager.save_temp_performance(agent_name, agent_summary)