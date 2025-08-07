"""
Storage backends for simulation data persistence.

This module provides various storage backends for saving and loading
simulation contexts, results, and metrics.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import yaml

from .context import SimulationContext
# Dict[str, Any] removed - use Dict[str, Any] for results


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_context(self, context: SimulationContext, identifier: str) -> None:
        """Save a simulation context."""
        pass
    
    @abstractmethod
    def load_context(self, identifier: str) -> Optional[SimulationContext]:
        """Load a simulation context."""
        pass
    
    @abstractmethod
    def save_result(self, result: Dict[str, Any], identifier: str) -> None:
        """Save a simulation result."""
        pass
    
    @abstractmethod
    def load_result(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load a simulation result."""
        pass
    
    @abstractmethod
    def list_simulations(self) -> List[str]:
        """List all stored simulation identifiers."""
        pass
    
    @abstractmethod
    def delete_simulation(self, identifier: str) -> bool:
        """Delete a stored simulation."""
        pass


class FileStorageBackend(StorageBackend):
    """File-based storage backend using JSON or pickle."""
    
    def __init__(
        self,
        storage_dir: Union[str, Path],
        format: str = "json",
    ):
        """
        Initialize file storage backend.
        
        Args:
            storage_dir: Directory to store files
            format: Storage format ("json" or "pickle")
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        
        if format not in ["json", "pickle"]:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_context_path(self, identifier: str) -> Path:
        """Get the file path for a context."""
        ext = "json" if self.format == "json" else "pkl"
        return self.storage_dir / f"{identifier}_context.{ext}"
    
    def _get_result_path(self, identifier: str) -> Path:
        """Get the file path for a result."""
        ext = "json" if self.format == "json" else "pkl"
        return self.storage_dir / f"{identifier}_result.{ext}"
    
    def save_context(self, context: SimulationContext, identifier: str) -> None:
        """Save a simulation context to file."""
        path = self._get_context_path(identifier)
        
        if self.format == "json":
            with open(path, "w") as f:
                json.dump(context.to_dict(), f, indent=2, default=str)
        else:
            with open(path, "wb") as f:
                pickle.dump(context, f)
    
    def load_context(self, identifier: str) -> Optional[SimulationContext]:
        """Load a simulation context from file."""
        path = self._get_context_path(identifier)
        
        if not path.exists():
            return None
        
        if self.format == "json":
            with open(path, "r") as f:
                data = json.load(f)
            # Reconstruct context from dictionary
            # This is a simplified version - full implementation would
            # properly reconstruct all nested objects
            context = SimulationContext(
                simulation_name=data["simulation_name"],
                start_time=datetime.fromisoformat(data["start_time"]),
            )
            context.state = data.get("state", {})
            context.metadata = data.get("metadata", {})
            context.metrics = data.get("metrics", {})
            context.errors = data.get("errors", [])
            if data.get("end_time"):
                context.end_time = datetime.fromisoformat(data["end_time"])
            return context
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def save_result(self, result: Dict[str, Any], identifier: str) -> None:
        """Save a simulation result to file."""
        path = self._get_result_path(identifier)
        
        if self.format == "json":
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        else:
            with open(path, "wb") as f:
                pickle.dump(result, f)
    
    def load_result(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load a simulation result from file."""
        path = self._get_result_path(identifier)
        
        if not path.exists():
            return None
        
        if self.format == "json":
            with open(path, "r") as f:
                data = json.load(f)
            # Reconstruct result from dictionary
            return data
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def list_simulations(self) -> List[str]:
        """List all stored simulation identifiers."""
        ext = "json" if self.format == "json" else "pkl"
        context_files = self.storage_dir.glob(f"*_context.{ext}")
        
        identifiers = []
        for file in context_files:
            # Extract identifier from filename
            identifier = file.stem.replace("_context", "")
            identifiers.append(identifier)
        
        return sorted(identifiers)
    
    def delete_simulation(self, identifier: str) -> bool:
        """Delete a stored simulation."""
        context_path = self._get_context_path(identifier)
        result_path = self._get_result_path(identifier)
        
        deleted = False
        if context_path.exists():
            context_path.unlink()
            deleted = True
        if result_path.exists():
            result_path.unlink()
            deleted = True
        
        return deleted


class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage backend."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize SQLite storage backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_contexts (
                    identifier TEXT PRIMARY KEY,
                    simulation_name TEXT,
                    data TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    identifier TEXT PRIMARY KEY,
                    simulation_name TEXT,
                    data TEXT,
                    overall_score REAL,
                    created_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_contexts_name 
                ON simulation_contexts(simulation_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_name 
                ON simulation_results(simulation_name)
            """)
    
    def save_context(self, context: SimulationContext, identifier: str) -> None:
        """Save a simulation context to database."""
        data = json.dumps(context.to_dict(), default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO simulation_contexts 
                (identifier, simulation_name, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                identifier,
                context.simulation_name,
                data,
                context.start_time,
                datetime.now()
            ))
    
    def load_context(self, identifier: str) -> Optional[SimulationContext]:
        """Load a simulation context from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT data FROM simulation_contexts 
                WHERE identifier = ?
            """, (identifier,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data = json.loads(row[0])
            
            # Reconstruct context
            context = SimulationContext(
                simulation_name=data["simulation_name"],
                start_time=datetime.fromisoformat(data["start_time"]),
            )
            context.state = data.get("state", {})
            context.metadata = data.get("metadata", {})
            context.metrics = data.get("metrics", {})
            context.errors = data.get("errors", [])
            if data.get("end_time"):
                context.end_time = datetime.fromisoformat(data["end_time"])
            
            return context
    
    def save_result(self, result: Dict[str, Any], identifier: str) -> None:
        """Save a simulation result to database."""
        data = json.dumps(result, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO simulation_results 
                (identifier, simulation_name, data, overall_score, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                identifier,
                result.simulation_name,
                data,
                result.evaluation.overall_score if result.evaluation else None,
                datetime.now()
            ))
    
    def load_result(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load a simulation result from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT data FROM simulation_results 
                WHERE identifier = ?
            """, (identifier,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data = json.loads(row[0])
            return data
    
    def list_simulations(self) -> List[str]:
        """List all stored simulation identifiers."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT identifier FROM simulation_contexts
                ORDER BY identifier
            """)
            
            return [row[0] for row in cursor.fetchall()]
    
    def delete_simulation(self, identifier: str) -> bool:
        """Delete a stored simulation."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete from both tables
            cursor1 = conn.execute("""
                DELETE FROM simulation_contexts WHERE identifier = ?
            """, (identifier,))
            
            cursor2 = conn.execute("""
                DELETE FROM simulation_results WHERE identifier = ?
            """, (identifier,))
            
            return (cursor1.rowcount + cursor2.rowcount) > 0
    
    def query_simulations(
        self,
        simulation_name: Optional[str] = None,
        min_score: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query simulations with filters.
        
        Args:
            simulation_name: Filter by simulation name
            min_score: Minimum overall score
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of simulation metadata
        """
        query = """
            SELECT 
                c.identifier,
                c.simulation_name,
                c.created_at,
                r.overall_score
            FROM simulation_contexts c
            LEFT JOIN simulation_results r ON c.identifier = r.identifier
            WHERE 1=1
        """
        params = []
        
        if simulation_name:
            query += " AND c.simulation_name = ?"
            params.append(simulation_name)
        
        if min_score is not None:
            query += " AND r.overall_score >= ?"
            params.append(min_score)
        
        if start_date:
            query += " AND c.created_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND c.created_at <= ?"
            params.append(end_date)
        
        query += " ORDER BY c.created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "identifier": row[0],
                    "simulation_name": row[1],
                    "created_at": row[2],
                    "overall_score": row[3],
                })
            
            return results


class StorageManager:
    """
    High-level storage manager supporting multiple backends.
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize storage manager.
        
        Args:
            backend: Storage backend to use (defaults to FileStorageBackend)
        """
        if backend is None:
            # Default to file storage in current directory
            backend = FileStorageBackend("./simulation_data")
        
        self.backend = backend
    
    def save_simulation(
        self,
        context: SimulationContext,
        result: Optional[Dict[str, Any]] = None,
        identifier: Optional[str] = None,
    ) -> str:
        """
        Save a complete simulation.
        
        Args:
            context: Simulation context
            result: Optional simulation result
            identifier: Optional identifier (auto-generated if not provided)
            
        Returns:
            The identifier used for storage
        """
        if identifier is None:
            # Generate identifier from simulation name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            identifier = f"{context.simulation_name}_{timestamp}"
        
        self.backend.save_context(context, identifier)
        
        if result:
            self.backend.save_result(result, identifier)
        
        return identifier
    
    def load_simulation(
        self,
        identifier: str,
    ) -> tuple[Optional[SimulationContext], Optional[Dict[str, Any]]]:
        """
        Load a complete simulation.
        
        Args:
            identifier: Simulation identifier
            
        Returns:
            Tuple of (context, result), either may be None if not found
        """
        context = self.backend.load_context(identifier)
        result = self.backend.load_result(identifier)
        
        return context, result
    
    def list_simulations(self) -> List[str]:
        """List all stored simulations."""
        return self.backend.list_simulations()
    
    def delete_simulation(self, identifier: str) -> bool:
        """Delete a stored simulation."""
        return self.backend.delete_simulation(identifier)
    
    def export_to_yaml(self, identifier: str, output_path: Union[str, Path]) -> bool:
        """
        Export a simulation to YAML format.
        
        Args:
            identifier: Simulation identifier
            output_path: Path to save YAML file
            
        Returns:
            True if successful, False otherwise
        """
        context, result = self.load_simulation(identifier)
        
        if not context:
            return False
        
        data = {
            "context": context.to_dict(),
        }
        
        if result:
            data["result"] = result
        
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return True


class JSONFileStorage:
    """
    Handles JSON file operations for storing metrics data.
    Uses atomic writes to prevent corruption.
    """
    
    def __init__(self, base_dir: Union[str, Path] = "./simulation_data"):
        """
        Initialize JSON file storage.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.tasks_dir = self.base_dir / "tasks"
        self.workflows_dir = self.base_dir / "workflows"
        self.history_dir = self.base_dir / "history"
        self.latest_file = self.base_dir / "latest_agent_performance.json"
        
        # Ensure directories exist
        self.tasks_dir.mkdir(exist_ok=True)
        self.workflows_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
    
    def _atomic_write(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Perform atomic write using temp file and rename.
        
        Args:
            file_path: Target file path
            data: Data to write
        """
        # Write to temp file first
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(file_path)
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def save_task_metrics(self, task_id: str, metrics: 'TaskMetrics') -> Path:
        """
        Save task metrics to JSON file.
        
        Args:
            task_id: Unique task identifier
            metrics: TaskMetrics object
            
        Returns:
            Path to saved file
        """
        from .models import TaskMetrics
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_id}_{timestamp}.json"
        file_path = self.tasks_dir / filename
        
        # Convert to dict and save
        data = metrics.model_dump()
        data['_saved_at'] = datetime.now().isoformat()
        self._atomic_write(file_path, data)
        
        return file_path
    
    def save_workflow_metrics(self, workflow_id: str, metrics: 'WorkflowMetrics') -> Path:
        """
        Save workflow metrics to JSON file.
        
        Args:
            workflow_id: Unique workflow identifier
            metrics: WorkflowMetrics object
            
        Returns:
            Path to saved file
        """
        from .models import WorkflowMetrics
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{workflow_id}_{timestamp}.json"
        file_path = self.workflows_dir / filename
        
        # Convert to dict and save
        data = metrics.model_dump()
        data['_saved_at'] = datetime.now().isoformat()
        self._atomic_write(file_path, data)
        
        return file_path
    
    def save_historical_performance(self, agent_name: str, metrics: 'TaskMetrics') -> Path:
        """
        Save agent historical performance data.
        
        Args:
            agent_name: Name of the agent
            metrics: TaskMetrics for the agent
            
        Returns:
            Path to saved file
        """
        from .models import TaskMetrics
        
        # Create agent directory if needed
        agent_dir = self.history_dir / agent_name
        agent_dir.mkdir(exist_ok=True)
        
        # Create filename with ISO timestamp
        timestamp = datetime.now().isoformat().replace(':', '-').replace('.', '-')
        filename = f"{timestamp}.json"
        file_path = agent_dir / filename
        
        # Save metrics with agent info
        data = {
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.model_dump()
        }
        self._atomic_write(file_path, data)
        
        # Update latest performance file
        self._update_latest_performance(agent_name, metrics)
        
        return file_path
    
    def _update_latest_performance(self, agent_name: str, metrics: 'TaskMetrics') -> None:
        """Update the latest performance file with new agent metrics."""
        # Load existing latest data
        latest_data = {}
        if self.latest_file.exists():
            try:
                with open(self.latest_file, 'r') as f:
                    latest_data = json.load(f)
            except json.JSONDecodeError:
                latest_data = {}
        
        # Update with new metrics
        latest_data[agent_name] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.model_dump()
        }
        
        # Save updated data
        self._atomic_write(self.latest_file, latest_data)
    
    def load_latest_performance(self) -> Dict[str, Any]:
        """
        Load the latest performance data for all agents.
        
        Returns:
            Dictionary mapping agent names to their latest metrics
        """
        if not self.latest_file.exists():
            return {}
        
        try:
            with open(self.latest_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def load_task_metrics(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load task metrics from a file.
        
        Args:
            file_path: Path to the metrics file
            
        Returns:
            Metrics data or None if not found
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def load_workflow_metrics(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load workflow metrics from a file.
        
        Args:
            file_path: Path to the metrics file
            
        Returns:
            Metrics data or None if not found
        """
        return self.load_task_metrics(file_path)  # Same format


class SQLiteIndexer:
    """
    Maintains a lightweight SQLite index for fast lookups of historical data.
    """
    
    def __init__(self, db_path: Union[str, Path] = "./simulation_data/index.db"):
        """
        Initialize SQLite indexer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Create performance index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    agent_name TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    tokens INTEGER NOT NULL,
                    duration INTEGER NOT NULL,
                    comment_score INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_id 
                ON performance_index(task_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_name 
                ON performance_index(agent_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_index(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_success 
                ON performance_index(success)
            """)
    
    def index_performance(self, task_id: str, file_path: Union[str, Path], 
                         metrics: 'TaskMetrics', agent_name: Optional[str] = None) -> None:
        """
        Index performance metrics for fast lookup.
        
        Args:
            task_id: Task identifier
            file_path: Path to the metrics file
            metrics: TaskMetrics object
            agent_name: Optional agent name
        """
        from .models import TaskMetrics
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_index 
                (task_id, agent_name, timestamp, file_path, success, 
                 tokens, duration, comment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                agent_name,
                datetime.now(),
                str(file_path),
                metrics.task_success,
                metrics.total_tokens,
                metrics.total_duration,
                metrics.comment_score
            ))
    
    def query_by_date_range(self, start: datetime, end: datetime) -> List[str]:
        """
        Query performance data by date range.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            List of file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT file_path 
                FROM performance_index 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (start, end))
            
            return [row[0] for row in cursor.fetchall()]
    
    def query_by_task_id(self, task_id: str) -> List[str]:
        """
        Query performance data by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT file_path 
                FROM performance_index 
                WHERE task_id = ?
                ORDER BY timestamp DESC
            """, (task_id,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def query_by_agent(self, agent_name: str, limit: Optional[int] = None) -> List[str]:
        """
        Query performance data by agent name.
        
        Args:
            agent_name: Agent name
            limit: Optional limit on results
            
        Returns:
            List of file paths
        """
        query = """
            SELECT DISTINCT file_path 
            FROM performance_index 
            WHERE agent_name = ?
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (agent_name,))
            return [row[0] for row in cursor.fetchall()]
    
    def get_performance_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregate performance statistics.
        
        Args:
            agent_name: Optional filter by agent
            
        Returns:
            Dictionary with performance statistics
        """
        base_query = """
            SELECT 
                COUNT(*) as total_tasks,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tasks,
                AVG(tokens) as avg_tokens,
                AVG(duration) as avg_duration,
                AVG(comment_score) as avg_score,
                MIN(timestamp) as first_run,
                MAX(timestamp) as last_run
            FROM performance_index
        """
        
        if agent_name:
            query = base_query + " WHERE agent_name = ?"
            params = (agent_name,)
        else:
            query = base_query
            params = ()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if not row or row[0] == 0:
                return {
                    'total_tasks': 0,
                    'successful_tasks': 0,
                    'success_rate': 0.0,
                    'avg_tokens': 0,
                    'avg_duration': 0,
                    'avg_score': 0,
                    'first_run': None,
                    'last_run': None
                }
            
            return {
                'total_tasks': row[0],
                'successful_tasks': row[1],
                'success_rate': row[1] / row[0] if row[0] > 0 else 0.0,
                'avg_tokens': round(row[2]) if row[2] else 0,
                'avg_duration': round(row[3]) if row[3] else 0,
                'avg_score': round(row[4], 1) if row[4] else 0,
                'first_run': row[5],
                'last_run': row[6]
            }