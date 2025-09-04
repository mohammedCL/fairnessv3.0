"""
Logging configuration settings for the Fairness Assessment Platform.
Centralized configuration for log levels, formats, and handlers.
"""

import os
from typing import Dict, Any

# Default logging configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment-based log level
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

# Development vs Production settings
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Log file settings
LOG_DIR = os.getenv("LOG_DIR", "logs")
MAX_LOG_FILE_SIZE = int(os.getenv("MAX_LOG_FILE_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Console logging settings
CONSOLE_LOG_LEVEL = "DEBUG" if DEBUG_MODE else "INFO"
CONSOLE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File logging settings  
FILE_LOG_LEVEL = "DEBUG"
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"

# JSON logging settings (for structured logs)
JSON_LOG_LEVEL = "INFO"

# Module-specific log levels
MODULE_LOG_LEVELS: Dict[str, str] = {
    "fairness.analysis": LOG_LEVEL,
    "fairness.upload": LOG_LEVEL,
    "fairness.mitigation": LOG_LEVEL,
    "fairness.api": LOG_LEVEL,
    "fairness.ai": LOG_LEVEL,
}

# Sensitive data fields to exclude from logs
SENSITIVE_FIELDS = [
    "password", "api_key", "secret", "token", 
    "authorization", "x-api-key", "bearer"
]

# Log message filters
EXCLUDE_PATTERNS = [
    "health check",  # Reduce noise from health checks
    "static file",   # Reduce noise from static file requests
]

def get_log_config() -> Dict[str, Any]:
    """Get complete logging configuration"""
    return {
        "log_level": LOG_LEVEL,
        "debug_mode": DEBUG_MODE,
        "log_dir": LOG_DIR,
        "max_file_size": MAX_LOG_FILE_SIZE,
        "backup_count": LOG_BACKUP_COUNT,
        "console": {
            "level": CONSOLE_LOG_LEVEL,
            "format": CONSOLE_LOG_FORMAT
        },
        "file": {
            "level": FILE_LOG_LEVEL,
            "format": FILE_LOG_FORMAT
        },
        "json": {
            "level": JSON_LOG_LEVEL
        },
        "module_levels": MODULE_LOG_LEVELS,
        "sensitive_fields": SENSITIVE_FIELDS,
        "exclude_patterns": EXCLUDE_PATTERNS
    }


def is_sensitive_field(field_name: str) -> bool:
    """Check if a field contains sensitive data"""
    field_lower = field_name.lower()
    return any(sensitive in field_lower for sensitive in SENSITIVE_FIELDS)


def should_exclude_message(message: str) -> bool:
    """Check if a log message should be excluded based on patterns"""
    message_lower = message.lower()
    return any(pattern in message_lower for pattern in EXCLUDE_PATTERNS)
