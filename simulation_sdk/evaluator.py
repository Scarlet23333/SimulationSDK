"""
LLM-based evaluation framework for simulation quality assessment.

This module provides interfaces and implementations for evaluating
task and workflow execution quality using LLMs and mock evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
import random
import json
import logging

# Models import removed - using simple dict/tuple returns instead

# Set up logging
logger = logging.getLogger(__name__)

# Try to import OpenAI, but make it optional
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. OpenAIEvaluator will fall back to MockEvaluator.")


class EvaluatorInterface(ABC):
    """Abstract base class for LLM evaluators."""
    
    @abstractmethod
    def evaluate_task(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluate a task execution and return a score with reasoning.
        
        Args:
            context: Dictionary containing:
                - task_goal: str - The goal/objective of the task
                - execution_trace: Dict - LangGraph execution data
                - tool_calls: List[Dict] - History of tool calls
                - final_output: Any - The task's final output
                - task_success: bool - Whether the task completed successfully
                - task_metrics: Optional[TaskMetrics] - Performance metrics
                
        Returns:
            Tuple of (score, reasoning) where:
                - score: float between 0.0 and 10.0
                - reasoning: str explaining the score
        """
        pass
    
    @abstractmethod
    def evaluate_workflow(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluate a workflow execution and return a score with reasoning.
        
        Args:
            context: Dictionary containing:
                - workflow_name: str - Name of the workflow
                - workflow_goal: str - Overall workflow objective
                - tasks: List[Dict] - Individual task contexts
                - workflow_success: bool - Whether workflow completed successfully
                - workflow_metrics: Optional[WorkflowMetrics] - Performance metrics
                
        Returns:
            Tuple of (score, reasoning) where:
                - score: float between 0.0 and 10.0
                - reasoning: str explaining the score
        """
        pass


class MockEvaluator(EvaluatorInterface):
    """Mock evaluator for testing purposes."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize mock evaluator with optional random seed."""
        if seed is not None:
            random.seed(seed)
    
    def evaluate_task(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Generate mock evaluation for tasks.
        
        Returns random scores:
        - 7-9 for successful tasks
        - 0-3 for failed tasks
        """
        task_success = context.get("task_success", True)
        task_name = context.get("task_name", context.get("agent_name", "Unknown task"))
        tool_calls = context.get("tool_calls", [])
        
        if task_success:
            # Successful tasks get scores between 7 and 9
            base_score = random.uniform(7.0, 9.0)
            
            # Adjust based on tool usage efficiency
            if tool_calls:
                unique_tools = len(set(call.get("tool_name", "") for call in tool_calls))
                efficiency_bonus = min(0.5, unique_tools * 0.1)
                score = min(10.0, base_score + efficiency_bonus)
            else:
                score = base_score
            
            reasoning = (
                f"Task '{task_name}' completed successfully. "
                f"Used {len(tool_calls)} tool calls efficiently. "
                f"Output quality appears satisfactory based on mock evaluation."
            )
        else:
            # Failed tasks get scores between 0 and 3
            score = random.uniform(0.0, 3.0)
            reasoning = (
                f"Task '{task_name}' failed to complete successfully. "
                f"Execution issues detected in the process. "
                f"Consider reviewing error handling and retry logic."
            )
        
        return round(score, 1), reasoning
    
    def evaluate_workflow(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Generate mock evaluation for workflows.
        
        Aggregates task scores with some variance.
        """
        workflow_name = context.get("workflow_name", "Unknown workflow")
        workflow_success = context.get("workflow_success", True)
        tasks = context.get("tasks", [])
        
        if not tasks:
            # No tasks means a poor workflow
            return 2.0, "Workflow contains no tasks. Unable to evaluate properly."
        
        # Evaluate each task and aggregate
        task_scores = []
        for task in tasks:
            # Handle TaskMetrics objects or dicts
            if hasattr(task, 'comment_score'):
                # TaskMetrics object - use existing score
                task_scores.append(task.comment_score)
            elif isinstance(task, dict) and 'comment_score' in task:
                # Dict with comment_score
                task_scores.append(task['comment_score'])
            else:
                # Need to evaluate - create context for evaluation
                task_context = task if isinstance(task, dict) else {"task_success": True}
                task_score, _ = self.evaluate_task(task_context)
                task_scores.append(task_score)
        
        # Calculate weighted average with some randomness
        avg_score = sum(task_scores) / len(task_scores)
        variance = random.uniform(-0.5, 0.5)
        workflow_score = max(0.0, min(10.0, avg_score + variance))
        
        if workflow_success and workflow_score >= 7.0:
            reasoning = (
                f"Workflow '{workflow_name}' executed successfully with {len(tasks)} tasks. "
                f"Average task performance was good. "
                f"Overall coordination and flow appears well-structured."
            )
        elif workflow_success:
            reasoning = (
                f"Workflow '{workflow_name}' completed but with room for improvement. "
                f"Some tasks underperformed. Consider optimizing task sequencing."
            )
        else:
            reasoning = (
                f"Workflow '{workflow_name}' failed to complete successfully. "
                f"Multiple task failures or coordination issues detected."
            )
        
        return round(workflow_score, 1), reasoning


class OpenAIEvaluator(EvaluatorInterface):
    """OpenAI-based evaluator for production use."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
        task_prompt_template: Optional[str] = None,
        workflow_prompt_template: Optional[str] = None,
        fallback_evaluator: Optional[EvaluatorInterface] = None
    ):
        """
        Initialize OpenAI evaluator.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
            model: OpenAI model to use for evaluation
            temperature: Sampling temperature for responses
            task_prompt_template: Custom prompt template for task evaluation
            workflow_prompt_template: Custom prompt template for workflow evaluation
            fallback_evaluator: Evaluator to use if OpenAI API fails
        """
        self.model = model
        self.temperature = temperature
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        elif OPENAI_AVAILABLE:
            # Try to use default client (will use OPENAI_API_KEY env var)
            try:
                self.client = openai.OpenAI()
            except Exception:
                self.client = None
        else:
            self.client = None
        
        # Set up prompt templates
        self.task_prompt_template = task_prompt_template or self._get_default_task_prompt()
        self.workflow_prompt_template = workflow_prompt_template or self._get_default_workflow_prompt()
        
        # Set up fallback evaluator
        self.fallback_evaluator = fallback_evaluator or MockEvaluator()
    
    def _format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format tool calls for the prompt."""
        if not tool_calls:
            return "No tool calls made"
        
        formatted = []
        for i, call in enumerate(tool_calls[:10]):  # Limit to first 10
            tool_name = call.get("tool_name", "unknown")
            formatted.append(f"{i+1}. {tool_name}")
        
        result = "\n".join(formatted)
        if len(tool_calls) > 10:
            result += f"\n... and {len(tool_calls) - 10} more tool calls"
        
        return result
    
    def _truncate_output(self, output: Any, max_length: int = 500) -> str:
        """Truncate output to avoid token limits."""
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, indent=2)
        else:
            output_str = str(output)
        
        if len(output_str) > max_length:
            return output_str[:max_length] + "\n... (truncated)"
        return output_str
    
    def _format_workflow_tasks(self, tasks: List[Dict[str, Any]], task_summaries: Dict[str, Any]) -> str:
        """Format workflow tasks for evaluation prompt."""
        formatted = []
        for i, task in enumerate(tasks[:10]):  # Limit to first 10
            task_name = task.get("task_name", f"Task {i+1}")
            task_score = task.get("comment_score", "N/A")
            formatted.append(f"- {task_name}: {task_score}/10")
        
        result = "\n".join(formatted)
        if len(tasks) > 10:
            result += f"\n... and {len(tasks) - 10} more tasks"
        
        return result
    
    def _get_default_task_prompt(self) -> str:
        """Get default prompt template for task evaluation."""
        return """You are an expert AI system evaluator. Evaluate this agent's performance on a scale of 0-10.

Agent: {agent_name}
Execution successful: {task_success}

Tool calls made by the agent:
{tool_calls_summary}

Final output produced:
{final_output}

Performance metrics:
- Tokens used: {tokens}
- Duration: {duration}ms

Evaluation criteria (in order of importance):
1. Correctness: Did the agent produce the expected output?
2. Efficiency: Did it use the minimum necessary tool calls?
3. Token usage: Was the token consumption reasonable?

Consider the specific nature of this agent when weighting these criteria.
For example, a research agent might justifiably use more tokens than a simple formatter.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 10.0>,
    "reasoning": "<brief explanation (2-3 sentences)>"
}}"""
    
    def _get_default_workflow_prompt(self) -> str:
        """Get default prompt template for workflow evaluation."""
        return """You are an expert AI system evaluator. Evaluate this workflow's overall performance on a scale of 0-10.

Workflow: {workflow_name}
Overall success: {workflow_success}

Tasks executed ({num_tasks} total):
{task_scores_summary}

Overall metrics:
- Total tokens: {total_tokens}
- Total duration: {total_duration}ms
- Estimated cost: ${total_cost:.4f}

Evaluation criteria:
1. Overall success: Did the workflow achieve its goal?
2. Task coordination: Were tasks executed in logical order?
3. Resource efficiency: Was the token/time usage reasonable for the complexity?

Weight the importance of each task based on its role in the workflow.
Critical path tasks should have more impact on the score than auxiliary ones.

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 10.0>,
    "reasoning": "<brief explanation (3-4 sentences)>"
}}"""
    
    def _call_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make a call to OpenAI API and parse the response."""
        if not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AI system evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            # Extract and parse JSON from response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.error("No JSON found in OpenAI response")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None
    
    def evaluate_task(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate task using OpenAI API with fallback to mock evaluator.
        
        Only evaluates AGENT_TOOL tasks. Other tool types return fixed scores.
        """
        # Check if this is an AGENT_TOOL evaluation
        agent_name = context.get("agent_name", "")
        agent_summary = context.get("agent_summary")
        
        if not agent_summary:
            # Not an AGENT_TOOL, return simple score
            task_success = context.get("task_success", False)
            return (10.0 if task_success else 0.0, "Non-agent tool")
        
        # Extract information from agent_summary
        task_success = context.get("task_success", False)
        
        if not task_success:
            return 0.0, "Task failed"
        
        # Get tool calls and final output from agent_summary
        tool_calls = agent_summary.get("tool_calls", [])
        final_output = agent_summary.get("final_output", {})
        performance = agent_summary.get("performance", {})
        
        # Summarize tool calls
        tool_calls_summary = self._format_tool_calls(tool_calls)
        
        # Format final output (truncate if too long)
        final_output_str = self._truncate_output(final_output, max_length=500)
        
        # Create prompt
        prompt = self.task_prompt_template.format(
            agent_name=agent_name,
            task_success=task_success,
            tool_calls_summary=tool_calls_summary,
            final_output=final_output_str,
            tokens=performance.get("tokens", 0),
            duration=performance.get("duration", 0)
        )
        
        # Try OpenAI evaluation
        result = self._call_openai(prompt)
        
        if result:
            score = float(result.get("score", 8.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            return score, reasoning
        else:
            # Fall back to mock evaluator
            logger.warning("OpenAI evaluation failed, using fallback evaluator")
            return self.fallback_evaluator.evaluate_task(context)
    
    def evaluate_workflow(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate workflow using OpenAI API with fallback to mock evaluator."""
        # Get workflow metrics from context
        workflow_metrics = context.get("workflow_metrics", {})
        task_execution_summaries = context.get("task_execution_summaries", {})
        
        # Extract workflow info
        workflow_name = workflow_metrics.get("workflow_name", "Unknown")
        workflow_success = workflow_metrics.get("workflow_success", False)
        tasks = workflow_metrics.get("tasks", [])
        total_tokens = workflow_metrics.get("total_tokens", 0)
        total_duration = workflow_metrics.get("total_duration", 0)
        total_cost = workflow_metrics.get("total_cost", 0.0)
        
        # Format task scores
        task_scores_summary = self._format_workflow_tasks(tasks, task_execution_summaries)
        
        # Create prompt
        prompt = self.workflow_prompt_template.format(
            workflow_name=workflow_name,
            workflow_success=workflow_success,
            num_tasks=len(tasks),
            task_scores_summary=task_scores_summary,
            total_tokens=total_tokens,
            total_duration=total_duration,
            total_cost=total_cost
        )
        
        # Try OpenAI evaluation
        result = self._call_openai(prompt)
        
        if result:
            score = float(result.get("score", 5.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            
            return score, reasoning
        else:
            # Fall back to mock evaluator
            logger.warning("OpenAI evaluation failed, using fallback evaluator")
            return self.fallback_evaluator.evaluate_workflow(context)


# Convenience function to create default evaluator
def create_evaluator(use_openai: bool = True, **kwargs) -> EvaluatorInterface:
    """
    Create an evaluator instance.
    
    Args:
        use_openai: Whether to use OpenAI evaluator (requires API key)
        **kwargs: Additional arguments passed to evaluator constructor
        
    Returns:
        EvaluatorInterface instance
    """
    if use_openai and OPENAI_AVAILABLE:
        return OpenAIEvaluator(**kwargs)
    else:
        if use_openai and not OPENAI_AVAILABLE:
            logger.warning("OpenAI requested but not available, using MockEvaluator")
        return MockEvaluator(**kwargs)