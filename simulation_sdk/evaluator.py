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
        task_goal = context.get("task_goal", "Unknown task")
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
                f"Task '{task_goal}' completed successfully. "
                f"Used {len(tool_calls)} tool calls efficiently. "
                f"Output quality appears satisfactory based on mock evaluation."
            )
        else:
            # Failed tasks get scores between 0 and 3
            score = random.uniform(0.0, 3.0)
            reasoning = (
                f"Task '{task_goal}' failed to complete successfully. "
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
            task_score, _ = self.evaluate_task(task)
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
    
    def _get_default_task_prompt(self) -> str:
        """Get default prompt template for task evaluation."""
        return """You are an expert AI system evaluator. Evaluate the following task execution on a scale of 0-10.

Task Goal: {task_goal}

Task Success: {task_success}

Tool Calls Made: {tool_calls_summary}

Execution Trace Summary:
{execution_summary}

Final Output:
{final_output}

Evaluation Criteria:
1. Goal Achievement (0-4 points): Did the task achieve its stated goal?
2. Efficiency (0-2 points): Were resources used efficiently? Minimal redundant tool calls?
3. Quality (0-2 points): Is the output high quality and complete?
4. Error Handling (0-2 points): Were errors handled gracefully?

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 10>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"]
}}"""
    
    def _get_default_workflow_prompt(self) -> str:
        """Get default prompt template for workflow evaluation."""
        return """You are an expert AI system evaluator. Evaluate the following workflow execution on a scale of 0-10.

Workflow Name: {workflow_name}
Workflow Goal: {workflow_goal}
Workflow Success: {workflow_success}

Number of Tasks: {num_tasks}
Task Success Rate: {task_success_rate}%

Individual Task Scores:
{task_scores_summary}

Evaluation Criteria:
1. Overall Goal Achievement (0-4 points): Did the workflow achieve its objective?
2. Task Coordination (0-2 points): Were tasks well-coordinated and sequenced?
3. Efficiency (0-2 points): Was the workflow efficient without redundancy?
4. Robustness (0-2 points): Did the workflow handle failures gracefully?

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0 and 10>,
    "reasoning": "<detailed explanation of the score>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"]
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
        """Evaluate task using OpenAI API with fallback to mock evaluator."""
        # Prepare context for prompt
        task_goal = context.get("task_goal", "Unknown")
        task_success = context.get("task_success", False)
        tool_calls = context.get("tool_calls", [])
        execution_trace = context.get("execution_trace", {})
        final_output = context.get("final_output", "No output")
        
        # Summarize tool calls
        tool_calls_summary = f"{len(tool_calls)} tool calls made"
        if tool_calls:
            tool_names = [call.get("tool_name", "unknown") for call in tool_calls[:5]]
            tool_calls_summary += f": {', '.join(tool_names)}"
            if len(tool_calls) > 5:
                tool_calls_summary += f", and {len(tool_calls) - 5} more"
        
        # Summarize execution trace
        execution_summary = json.dumps(execution_trace, indent=2)[:500]
        if len(json.dumps(execution_trace)) > 500:
            execution_summary += "\n... (truncated)"
        
        # Format final output
        if isinstance(final_output, (dict, list)):
            final_output_str = json.dumps(final_output, indent=2)[:500]
        else:
            final_output_str = str(final_output)[:500]
        
        # Create prompt
        prompt = self.task_prompt_template.format(
            task_goal=task_goal,
            task_success=task_success,
            tool_calls_summary=tool_calls_summary,
            execution_summary=execution_summary,
            final_output=final_output_str
        )
        
        # Try OpenAI evaluation
        result = self._call_openai(prompt)
        
        if result:
            score = float(result.get("score", 5.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Optionally create full EvaluationResult
            if "strengths" in result:
                evaluation_result = EvaluationResult(
                    comment_score=int(score),
                    reasoning=reasoning,
                    strengths=result.get("strengths", []),
                    weaknesses=result.get("weaknesses", []),
                    suggestions=result.get("suggestions", [])
                )
                # Store in context for later use
                context["evaluation_result"] = evaluation_result
            
            return score, reasoning
        else:
            # Fall back to mock evaluator
            logger.warning("OpenAI evaluation failed, using fallback evaluator")
            return self.fallback_evaluator.evaluate_task(context)
    
    def evaluate_workflow(self, context: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate workflow using OpenAI API with fallback to mock evaluator."""
        # Prepare context for prompt
        workflow_name = context.get("workflow_name", "Unknown")
        workflow_goal = context.get("workflow_goal", "Unknown objective")
        workflow_success = context.get("workflow_success", False)
        tasks = context.get("tasks", [])
        
        # Calculate task statistics
        num_tasks = len(tasks)
        successful_tasks = sum(1 for task in tasks if task.get("task_success", False))
        task_success_rate = (successful_tasks / num_tasks * 100) if num_tasks > 0 else 0
        
        # Get individual task scores (if available)
        task_scores_summary = []
        for i, task in enumerate(tasks):
            task_name = task.get("task_goal", f"Task {i+1}")
            if "comment_score" in task:
                task_scores_summary.append(f"- {task_name}: {task['comment_score']}/10")
            else:
                # Evaluate task if not already evaluated
                score, _ = self.evaluate_task(task)
                task_scores_summary.append(f"- {task_name}: {score}/10")
        
        task_scores_str = "\n".join(task_scores_summary[:10])  # Limit to 10 tasks
        if len(tasks) > 10:
            task_scores_str += f"\n... and {len(tasks) - 10} more tasks"
        
        # Create prompt
        prompt = self.workflow_prompt_template.format(
            workflow_name=workflow_name,
            workflow_goal=workflow_goal,
            workflow_success=workflow_success,
            num_tasks=num_tasks,
            task_success_rate=task_success_rate,
            task_scores_summary=task_scores_str
        )
        
        # Try OpenAI evaluation
        result = self._call_openai(prompt)
        
        if result:
            score = float(result.get("score", 5.0))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Optionally create full EvaluationResult
            if "strengths" in result:
                evaluation_result = EvaluationResult(
                    comment_score=int(score),
                    reasoning=reasoning,
                    strengths=result.get("strengths", []),
                    weaknesses=result.get("weaknesses", []),
                    suggestions=result.get("suggestions", [])
                )
                # Store in context for later use
                context["evaluation_result"] = evaluation_result
            
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