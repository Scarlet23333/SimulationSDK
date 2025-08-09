"""
Performance data management for agent simulations.

This module provides a simple manager for storing and retrieving agent performance
data with explicit agent names and a two-stage update process.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .models import AgentAsToolSummary, TaskMetrics, WorkflowMetrics


class PerformanceManager:
    """
    Manages agent performance data with a two-stage update process.
    
    This class handles:
    - Storing temporary performance data for concurrent writes
    - Merging temporary files into a single latest performance file
    - Maintaining historical records with timestamps
    - Loading agent performance data for reuse
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the PerformanceManager.
        
        Args:
            base_path: Base directory for performance data storage. 
                      If None, uses SIMULATION_STORAGE_PATH env var or defaults to "simulation_data"
        """
        import os
        if base_path is None:
            base_path = os.environ.get("SIMULATION_STORAGE_PATH", "simulation_data")
        self.base_path = Path(base_path)
        self.temp_dir = self.base_path / "temp_performance"
        self.latest_file = self.base_path / "latest_agent_performance.json"
        self.history_dir = self.base_path / "history"
        
        # Create necessary directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def save_temp_performance(self, agent_name: str, summary: AgentAsToolSummary) -> None:
        """
        Save agent performance data to a temporary file.
        
        Args:
            agent_name: Explicit name of the agent
            summary: Agent performance summary including delay configuration
        """
        temp_file = self.temp_dir / f"{agent_name}.json"
        
        try:
            # Write summary directly to temporary file
            with open(temp_file, 'w') as f:
                json.dump(summary.model_dump(), f, indent=2)
            
            self.logger.info(f"Saved temporary performance data for {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save temp performance for {agent_name}: {e}")
            raise
    
    def merge_temp_to_latest(self) -> None:
        """
        Merge all temporary performance files into the latest performance file.
        
        This method:
        1. Reads all temp files
        2. Merges them into a single latest file with {agent_name: AgentAsToolSummary} structure
        3. Clears temp files after successful merge
        """
        try:
            # Load existing latest data if it exists
            existing_data = {}
            if self.latest_file.exists():
                with open(self.latest_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Collect all performance data from temp files
            merged_data = existing_data.copy()
            
            for temp_file in self.temp_dir.glob("*.json"):
                try:
                    # Extract agent name from filename
                    agent_name = temp_file.stem
                    
                    # Load temp file data
                    with open(temp_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Update merged data
                    merged_data[agent_name] = summary_data
                    
                except Exception as e:
                    self.logger.error(f"Failed to read temp file {temp_file}: {e}")
                    continue
            
            # Write merged data to latest file
            if merged_data:
                with open(self.latest_file, 'w') as f:
                    json.dump(merged_data, f, indent=2)
                
                self.logger.info(f"Merged {len(merged_data)} agent performances to latest file")
                
                # Clear temp files after successful merge
                for temp_file in self.temp_dir.glob("*.json"):
                    temp_file.unlink()
                    
                self.logger.info("Cleared temporary performance files")
            else:
                self.logger.warning("No performance data to merge")
                
        except Exception as e:
            self.logger.error(f"Failed to merge temp files: {e}")
            raise
    
    def load_agent_performance(self, agent_name: str) -> Optional[AgentAsToolSummary]:
        """
        Load agent performance data from the latest file.
        
        Args:
            agent_name: Explicit name of the agent
            
        Returns:
            AgentAsToolSummary if found, None otherwise
        """
        try:
            # Check if latest file exists
            if not self.latest_file.exists():
                self.logger.warning(f"Latest performance file not found")
                return None
            
            # Load latest data
            with open(self.latest_file, 'r') as f:
                latest_data = json.load(f)
            
            # Get agent data by explicit name
            if agent_name in latest_data:
                summary_data = latest_data[agent_name]
                return AgentAsToolSummary(**summary_data)
            else:
                self.logger.warning(f"No performance data found for agent: {agent_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load performance for {agent_name}: {e}")
            return None
    
    def save_task_history(self, agent_name: str, metrics: TaskMetrics) -> None:
        """
        Save full task metrics to historical storage.
        
        Args:
            agent_name: Explicit name of the agent
            metrics: Full task metrics including all tool calls
        """
        try:
            # Create agent history directory
            agent_history_dir = self.history_dir / agent_name
            agent_history_dir.mkdir(exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            history_file = agent_history_dir / f"{timestamp}.json"
            
            # Write task metrics
            with open(history_file, 'w') as f:
                json.dump(metrics.model_dump(), f, indent=2)
            
            self.logger.info(f"Saved task history for {agent_name} at {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Failed to save task history for {agent_name}: {e}")
            raise
    
    def save_workflow_metrics(self, workflow_metrics: WorkflowMetrics) -> Path:
        """
        Save workflow metrics to history storage.
        
        Args:
            workflow_metrics: WorkflowMetrics instance to save
            
        Returns:
            Path to the saved file
        """
        try:
            # Create directory structure
            workflow_history_dir = self.history_dir / "workflows" / workflow_metrics.workflow_id
            workflow_history_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            filename = f"{timestamp}.json"
            filepath = workflow_history_dir / filename
            
            # Save workflow metrics
            with open(filepath, 'w') as f:
                json.dump(workflow_metrics.model_dump(), f, indent=2, default=str)
            
            self.logger.info(f"Saved workflow metrics for {workflow_metrics.workflow_name} to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow metrics: {e}")
            raise