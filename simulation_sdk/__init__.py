"""
SimulationSDK - A comprehensive Python framework for creating and evaluating conversational simulations.

This package provides tools for building simulated conversational agents,
managing dialogue flows, and evaluating simulation performance.
"""

__version__ = "0.1.0"
__author__ = "SimulationSDK Team"

# Core decorators
from .decorators import simulation_tool, simulation_agent

# Models
from .models import ToolCategory, SimulationResponse

# Testing utilities
from .testing import test_connectivity

# Registry
from .registry import ToolRegistry

# Performance management utility
from .performance import PerformanceManager

# Context and workflow utilities
from .context import get_current_context


def merge_performance_files() -> None:
    """
    Merge all temporary performance files into the latest performance file.
    
    This is a CLI utility for manual performance management.
    """
    manager = PerformanceManager()
    manager.merge_temp_to_latest()


def save_workflow_metrics(workflow_metrics, base_path: str = "simulation_data"):
    """
    Save workflow metrics to history storage.
    
    This is a convenience function that uses PerformanceManager.
    
    Args:
        workflow_metrics: WorkflowMetrics instance to save
        base_path: Base directory for simulation data
        
    Returns:
        Path to the saved file
    """
    manager = PerformanceManager(base_path)
    return manager.save_workflow_metrics(workflow_metrics)


__all__ = [
    # Decorators
    "simulation_tool",
    "simulation_agent",
    
    # Models
    "ToolCategory",
    "SimulationResponse",
    
    # Testing
    "test_connectivity",
    
    # Registry
    "ToolRegistry",
    
    # Context and workflow
    "get_current_context",
    "save_workflow_metrics",
    
    # Manual performance management (optional)
    "merge_performance_files",
]