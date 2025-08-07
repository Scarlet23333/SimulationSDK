"""
Tool registry system for managing tool categorization and simulation responses.

This module provides a centralized registry for storing tool categories,
response templates, and handles template substitution with realistic delays
and error injection for production-affecting tools in simulation mode.
"""

import os
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from string import Template
from .models import ToolCategory, SimulationResponse


class ToolRegistry:
    """
    Singleton registry for managing tool categories and response templates.
    
    This class maintains a global registry of tools, their categories,
    and response templates for simulation mode.
    """
    
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance
    
    def register(
        self,
        name: str,
        category: ToolCategory,
        success_template: Optional[SimulationResponse] = None,
        failure_template: Optional[SimulationResponse] = None,
        description: str = "",
        connectivity_test_endpoint: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> None:
        """
        Register a tool with its category and response templates.
        
        Args:
            name: Tool name
            category: Tool category (READ_ONLY, PRODUCTION_AFFECTING, AGENT_TOOL)
            success_template: Template for successful responses in simulation mode
            failure_template: Template for failure responses in simulation mode
            description: Tool description
            connectivity_test_endpoint: Optional endpoint for connectivity testing
            agent_name: Optional agent name for AGENT_TOOL category
        """
        entry = {
            "name": name,
            "category": category,
            "description": description,
            "success_template": success_template,
            "failure_template": failure_template,
            "connectivity_test_endpoint": connectivity_test_endpoint,
            "agent_name": agent_name,
        }
        self._tools[name] = entry
    
    def get_category(self, name: str) -> Optional[ToolCategory]:
        """
        Get the category of a registered tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool category or None if not registered
        """
        entry = self._tools.get(name)
        return entry["category"] if entry else None
    
    def get_templates(self, name: str) -> Tuple[Optional[SimulationResponse], Optional[SimulationResponse]]:
        """
        Get response templates for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tuple of (success_template, failure_template)
        """
        entry = self._tools.get(name)
        if entry:
            return entry["success_template"], entry["failure_template"]
        return None, None
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool name
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._tools
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        List registered tool names, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category is None:
            return list(self._tools.keys())
        
        return [
            name for name, entry in self._tools.items()
            if entry["category"] == category
        ]
    
    def get_entry(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the full registry entry for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool entry dictionary or None if not registered
        """
        return self._tools.get(name)
    
    def get_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information including category and agent_name.
        
        Args:
            name: Tool name
            
        Returns:
            Dictionary with tool info or None if not registered
        """
        entry = self._tools.get(name)
        if not entry:
            return None
        
        return {
            "name": entry["name"],
            "category": entry["category"],
            "description": entry["description"],
            "agent_name": entry.get("agent_name"),
        }
    
    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._tools.clear()


class ResponseTemplateManager:
    """
    Manages response template substitution with realistic delays and error injection.
    
    This class handles the generation of simulated responses for production-affecting
    tools, including variable substitution, delay simulation, and error injection
    based on the SIMULATION_ERROR_RATE environment variable.
    """
    
    def __init__(self):
        """Initialize the template manager."""
        self.error_rate = float(os.environ.get("SIMULATION_ERROR_RATE", "0.0"))
        self.min_delay = 0.1  # 100ms minimum delay
        self.max_delay = 2.0  # 2s maximum delay
    
    def apply_template(
        self,
        template: SimulationResponse,
        variables: Dict[str, Any],
        tool_name: str,
    ) -> SimulationResponse:
        """
        Apply a template with variable substitution and realistic delays.
        
        Args:
            template: SimulationResponse template
            variables: Variables to substitute in the template
            tool_name: Name of the tool (for logging)
            
        Returns:
            Processed SimulationResponse object
        """
        # Add realistic delay
        delay = self._calculate_delay(tool_name)
        time.sleep(delay)
        
        # Process response data with variable substitution
        processed_data = self.substitute_variables(
            template.response_data,
            variables
        )
        
        # Process error message if present
        processed_error_message = None
        if template.error_message is not None:
            processed_error_message = self._substitute_string(
                template.error_message,
                variables
            )
        
        # Create new SimulationResponse with processed data
        return SimulationResponse(
            success=template.success,
            response_data=processed_data,
            error_code=template.error_code,
            error_message=processed_error_message
        )
    
    def should_inject_error(self) -> bool:
        """
        Determine if an error should be injected based on error rate.
        
        Returns:
            True if error should be injected
        """
        return random.random() < self.error_rate
    
    def _calculate_delay(self, tool_name: str) -> float:
        """
        Calculate a realistic delay for the tool response.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Delay in seconds
        """
        # Use tool name hash to get consistent but varied delays
        base_delay = hash(tool_name) % 1000 / 1000.0
        
        # Scale to desired range with some randomness
        delay = self.min_delay + (base_delay * (self.max_delay - self.min_delay))
        
        # Add 10% random variation
        variation = delay * 0.1 * (random.random() - 0.5)
        
        return max(self.min_delay, delay + variation)
    
    def substitute_variables(
        self,
        data: Any,
        variables: Dict[str, Any]
    ) -> Any:
        """
        Recursively substitute variables in data structure.
        
        Args:
            data: Data structure containing template strings
            variables: Variables to substitute
            
        Returns:
            Data with substituted variables
        """
        if isinstance(data, str):
            return self._substitute_string(data, variables)
        elif isinstance(data, dict):
            return {
                key: self.substitute_variables(value, variables)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self.substitute_variables(item, variables)
                for item in data
            ]
        else:
            return data
    
    def _substitute_string(self, template_str: str, variables: Dict[str, Any]) -> str:
        """
        Substitute variables in a template string.
        
        Args:
            template_str: Template string with $variable placeholders
            variables: Variables to substitute
            
        Returns:
            String with substituted variables
        """
        # Use Python's Template for safe substitution
        template = Template(template_str)
        
        # Convert all variables to strings for substitution
        str_variables = {k: str(v) for k, v in variables.items()}
        
        try:
            return template.safe_substitute(str_variables)
        except Exception:
            # Return original string if substitution fails
            return template_str