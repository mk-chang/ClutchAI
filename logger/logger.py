"""
Simple logging configuration for ClutchAI.

This module provides a simple, best-practice logging setup that:
- Uses module-level loggers (one per module)
- Supports debug mode via environment variable or parameter
- Provides consistent formatting
- Keeps it simple and not over-engineered
"""

import logging
import os
import sys
from typing import Optional


class Logger:
    """
    Logger class for ClutchAI package.
    
    This class provides a centralized logging configuration and management.
    It wraps Python's logging module with ClutchAI-specific defaults.
    """
    
    _configured = False
    _root_logger_name = "ClutchAI"
    
    @classmethod
    def setup(cls, debug: bool = False, level: Optional[int] = None) -> None:
        """
        Configure logging for the ClutchAI package.
        
        Args:
            debug: Enable debug mode (sets level to DEBUG)
            level: Optional logging level (overrides debug if provided)
        """
        # Determine logging level
        if level is not None:
            log_level = level
        elif debug:
            log_level = logging.DEBUG
        else:
            # Check environment variable as fallback
            debug_env = os.environ.get("CLUTCHAI_DEBUG", "").lower() in ("1", "true", "yes")
            log_level = logging.DEBUG if debug_env else logging.INFO
        
        # Configure root logger for ClutchAI package
        logger = logging.getLogger(cls._root_logger_name)
        logger.setLevel(log_level)
        
        # Avoid adding multiple handlers if already configured
        if logger.handlers:
            cls._configured = True
            return
        
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)  # Use stderr for logs (best practice)
        handler.setLevel(log_level)
        
        # Create formatter
        # Simple format: LEVEL - message
        # For debug: include module and function name
        if log_level == logging.DEBUG:
            formatter = logging.Formatter(
                '%(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Best practice: Use module-level loggers, one per module.
        Usage: logger = Logger.get_logger(__name__)
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Logger instance
        """
        # Ensure logging is set up (idempotent)
        if not cls._configured:
            cls.setup()
        
        # Return logger with ClutchAI prefix
        return logging.getLogger(f"{cls._root_logger_name}.{name}")


# Backward compatibility: Keep function-based API
def setup_logging(debug: bool = False, level: Optional[int] = None) -> None:
    """
    Configure logging for the ClutchAI package.
    
    This is a convenience function that wraps Logger.setup().
    
    Args:
        debug: Enable debug mode (sets level to DEBUG)
        level: Optional logging level (overrides debug if provided)
    """
    Logger.setup(debug=debug, level=level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    This is a convenience function that wraps Logger.get_logger().
    
    Best practice: Use module-level loggers, one per module.
    Usage: logger = get_logger(__name__)
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return Logger.get_logger(name)

