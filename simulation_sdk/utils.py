"""
Utility functions and helpers for the SimulationSDK.

This module provides various utility functions used throughout
the SDK, including logging, validation, and common operations.
"""

import os
import sys
import json
import yaml
import logging
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar
from datetime import datetime
from functools import wraps
import time
import re
from contextlib import contextmanager


# Type variable for generic functions
T = TypeVar('T')


# Configure logging
def setup_logging(
    level: Union[int, str] = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        format: Log message format
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("simulation_sdk")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)
    
    return logger


# Default logger
logger = setup_logging()


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager to time operations.
    
    Args:
        name: Name of the operation being timed
        
    Yields:
        Start time
    """
    start = time.time()
    logger.debug(f"{name} started")
    
    try:
        yield start
    finally:
        duration = time.time() - start
        logger.info(f"{name} completed in {duration:.3f} seconds")


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True
    except ImportError:
        logger.warning("jsonschema not installed, skipping validation")
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Schema validation failed: {str(e)}")
        return False


def load_config(
    config_path: Union[str, Path],
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        format: File format (auto-detected if None)
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect format from extension
    if format is None:
        format = config_path.suffix.lower().lstrip('.')
    
    with open(config_path, 'r') as f:
        if format == 'json':
            return json.load(f)
        elif format in ['yaml', 'yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {format}")


def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format (auto-detected if None)
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from extension
    if format is None:
        format = config_path.suffix.lower().lstrip('.')
    
    with open(config_path, 'w') as f:
        if format == 'json':
            json.dump(config, f, indent=2, default=str)
        elif format in ['yaml', 'yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported configuration format: {format}")


def generate_id(prefix: str = "sim", length: int = 8) -> str:
    """
    Generate a unique identifier.
    
    Args:
        prefix: Prefix for the ID
        length: Length of random part
        
    Returns:
        Generated ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.sha256(
        f"{timestamp}{time.time()}".encode()
    ).hexdigest()[:length]
    
    return f"{prefix}_{timestamp}_{random_part}"


def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized or "unnamed"


def deep_merge(
    dict1: Dict[str, Any],
    dict2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (values override dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Key separator
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(
                flatten_dict(value, new_key, separator).items()
            )
        else:
            items.append((new_key, value))
    
    return dict(items)


def extract_variables(
    text: str,
    pattern: str = r"\{(\w+)\}",
) -> List[str]:
    """
    Extract variable names from text.
    
    Args:
        text: Text containing variables
        pattern: Regex pattern for variables
        
    Returns:
        List of variable names
    """
    return re.findall(pattern, text)


def substitute_variables(
    text: str,
    variables: Dict[str, Any],
    pattern: str = r"\{(\w+)\}",
) -> str:
    """
    Substitute variables in text.
    
    Args:
        text: Text containing variables
        variables: Variable values
        pattern: Regex pattern for variables
        
    Returns:
        Text with substituted variables
    """
    def replacer(match):
        var_name = match.group(1)
        return str(variables.get(var_name, match.group(0)))
    
    return re.sub(pattern, replacer, text)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(
        self,
        calls_per_second: float = 1.0,
        burst: int = 1,
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second
            burst: Maximum burst size
        """
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on elapsed time
        self.tokens = min(
            self.burst,
            self.tokens + elapsed * self.calls_per_second
        )
        self.last_update = now
        
        # Wait if not enough tokens
        wait_time = 0.0
        if self.tokens < tokens:
            wait_time = (tokens - self.tokens) / self.calls_per_second
            time.sleep(wait_time)
            self.tokens = 0
        else:
            self.tokens -= tokens
        
        return wait_time


def create_progress_callback(
    total: int,
    prefix: str = "Progress",
    width: int = 50,
) -> Callable[[int], None]:
    """
    Create a progress callback function.
    
    Args:
        total: Total number of items
        prefix: Progress bar prefix
        width: Progress bar width
        
    Returns:
        Progress callback function
    """
    def callback(current: int) -> None:
        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        
        sys.stdout.write(f"\r{prefix}: |{bar}| {percent:.1%}")
        sys.stdout.flush()
        
        if current >= total:
            sys.stdout.write("\n")
    
    return callback


def safe_json_loads(
    text: str,
    default: Any = None,
) -> Any:
    """
    Safely load JSON with error handling.
    
    Args:
        text: JSON text
        default: Default value on error
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {str(e)}")
        return default


def format_exception(
    e: Exception,
    include_traceback: bool = True,
) -> str:
    """
    Format an exception for logging.
    
    Args:
        e: Exception to format
        include_traceback: Whether to include full traceback
        
    Returns:
        Formatted exception string
    """
    parts = [f"{type(e).__name__}: {str(e)}"]
    
    if include_traceback:
        parts.append("\nTraceback:")
        parts.append(traceback.format_exc())
    
    return "\n".join(parts)


# Environment utilities
def get_env_bool(
    key: str,
    default: bool = False,
) -> bool:
    """
    Get boolean value from environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        
    Returns:
        Boolean value
    """
    value = os.environ.get(key, "").lower()
    
    if value in ["true", "1", "yes", "on"]:
        return True
    elif value in ["false", "0", "no", "off"]:
        return False
    else:
        return default


def get_env_list(
    key: str,
    separator: str = ",",
    default: Optional[List[str]] = None,
) -> List[str]:
    """
    Get list value from environment variable.
    
    Args:
        key: Environment variable name
        separator: List separator
        default: Default value
        
    Returns:
        List of values
    """
    value = os.environ.get(key, "")
    
    if not value:
        return default or []
    
    return [item.strip() for item in value.split(separator) if item.strip()]