"""
Centralized logging configuration for the Fairness Assessment Platform.
Provides structured logging with different levels and formatters.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'job_id'):
            log_entry['job_id'] = record.job_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        if hasattr(record, 'bias_metrics_count'):
            log_entry['bias_metrics_count'] = record.bias_metrics_count
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class FairnessLogger:
    """Centralized logger for the Fairness Assessment Platform"""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.log_dir = "logs"
        self._setup_log_directory()
        self._setup_formatters()
    
    def _setup_log_directory(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _setup_formatters(self):
        """Setup different formatters for different use cases"""
        # Console formatter - simple and readable
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File formatter - more detailed
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # JSON formatter for structured logging
        self.json_formatter = JSONFormatter()
    
    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        """Get or create a logger with the specified name and level"""
        
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler - for development and immediate feedback
        # Use UTF-8 encoding for Windows compatibility with emojis
        try:
            # Try to configure stdout encoding to UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
        except:
            # If reconfigure fails, continue without it
            pass
            
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        
        # File handler - for persistent logging
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, f"{name}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.file_formatter)
        
        # JSON handler - for structured analysis
        json_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, f"{name}_structured.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(self.json_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(json_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self._loggers[name] = logger
        return logger


# Convenience functions for getting loggers
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a logger instance"""
    return FairnessLogger().get_logger(name, level)


def get_analysis_logger() -> logging.Logger:
    """Get logger for analysis operations"""
    return get_logger("fairness.analysis")


def get_upload_logger() -> logging.Logger:
    """Get logger for upload operations"""
    return get_logger("fairness.upload")


def get_mitigation_logger() -> logging.Logger:
    """Get logger for mitigation operations"""
    return get_logger("fairness.mitigation")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations"""
    return get_logger("fairness.api")


def get_ai_logger() -> logging.Logger:
    """Get logger for AI recommendation operations"""
    return get_logger("fairness.ai")


# Context manager for timed operations
class LogExecutionTime:
    """Context manager to log execution time of operations"""
    
    def __init__(self, logger: logging.Logger, operation_name: str, job_id: Optional[str] = None):
        self.logger = logger
        self.operation_name = operation_name
        self.job_id = job_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        extra = {"job_id": self.job_id} if self.job_id else {}
        self.logger.info(f"Starting {self.operation_name}", extra=extra)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (datetime.utcnow() - self.start_time).total_seconds()
        extra = {
            "execution_time": execution_time,
            "job_id": self.job_id
        } if self.job_id else {"execution_time": execution_time}
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation_name} in {execution_time:.2f}s", 
                extra=extra
            )
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {execution_time:.2f}s: {exc_val}", 
                extra=extra,
                exc_info=True
            )


# Decorators for common logging patterns
def log_function_entry_exit(logger: logging.Logger):
    """Decorator to log function entry and exit"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_bias_detection_metrics(logger: logging.Logger):
    """Decorator to log bias detection metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Extract metrics count if result is a list of BiasMetric objects
                metrics_count = len(result) if isinstance(result, list) else 0
                
                logger.info(
                    f"Bias detection completed: {metrics_count} metrics generated",
                    extra={
                        "execution_time": execution_time,
                        "bias_metrics_count": metrics_count
                    }
                )
                return result
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    f"Bias detection failed after {execution_time:.2f}s: {str(e)}",
                    extra={"execution_time": execution_time},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator
