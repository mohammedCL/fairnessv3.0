# Logging System Documentation

## Overview

The Fairness Assessment Platform now includes a comprehensive, production-ready logging system that replaces all print statements with structured, configurable logging.

## Features

### ✅ **Structured Logging**
- **JSON Format**: Machine-readable logs for analysis
- **Context Fields**: Job IDs, execution times, metrics counts
- **Multiple Formats**: Console (human-readable) + File (detailed) + JSON (structured)

### ✅ **Module-Specific Loggers**
- `fairness.api` - API endpoint operations
- `fairness.analysis` - Bias detection and analysis
- `fairness.upload` - File upload operations
- `fairness.mitigation` - Bias mitigation processes
- `fairness.ai` - AI recommendation services

### ✅ **Log Levels & Configuration**
- **DEBUG**: Detailed information for debugging
- **INFO**: General operational messages
- **WARNING**: Warning conditions
- **ERROR**: Error conditions with full stack traces

### ✅ **Production Features**
- **File Rotation**: Automatic log rotation (10MB max, 5 backups)
- **Environment Configuration**: Configurable via environment variables
- **Performance Monitoring**: Execution time tracking
- **Error Tracking**: Full exception details with context

## Quick Start

### Basic Usage
```python
from app.utils.logger import get_analysis_logger

logger = get_analysis_logger()

# Basic logging
logger.info("Analysis started")
logger.error("Analysis failed", exc_info=True)

# Structured logging with context
logger.info(
    "Bias detection completed", 
    extra={"job_id": "abc-123", "bias_metrics_count": 42}
)
```

### Execution Time Tracking
```python
from app.utils.logger import LogExecutionTime, get_analysis_logger

logger = get_analysis_logger()

with LogExecutionTime(logger, "bias detection", "job-456"):
    # Your analysis code here
    perform_bias_detection()
    # Automatically logs start time, end time, and duration
```

### Error Logging
```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "Operation failed", 
        exc_info=True,  # Includes full stack trace
        extra={"job_id": job_id, "operation": "bias_detection"}
    )
```

## Log File Structure

The logging system creates organized log files in the `logs/` directory:

```
logs/
├── fairness.api.log              # Human-readable API logs
├── fairness.api_structured.log   # JSON API logs
├── fairness.analysis.log         # Human-readable analysis logs
├── fairness.analysis_structured.log  # JSON analysis logs
├── fairness.upload.log           # Human-readable upload logs
├── fairness.upload_structured.log    # JSON upload logs
├── fairness.mitigation.log       # Human-readable mitigation logs
├── fairness.mitigation_structured.log  # JSON mitigation logs
├── fairness.ai.log               # Human-readable AI logs
└── fairness.ai_structured.log    # JSON AI logs
```

## Configuration

### Environment Variables
```bash
# Log level configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
DEBUG=false                       # Enable debug mode

# Log file settings
LOG_DIR=logs                      # Log directory path
MAX_LOG_FILE_SIZE=10485760       # 10MB max file size
LOG_BACKUP_COUNT=5               # Number of backup files
```

### JSON Log Format Example
```json
{
  "timestamp": "2025-09-04T10:06:43.627488",
  "level": "INFO",
  "module": "analysis_service",
  "function": "_detect_bias",
  "line": 950,
  "message": "Bias detection completed: 42 metrics generated",
  "job_id": "analysis-job-123",
  "bias_metrics_count": 42,
  "execution_time": 15.23
}
```

## What Changed

### ❌ **Before (Print Statements)**
```python
print(f"Starting analysis for job {job_id}")
print(f"Found {len(sensitive_features)} sensitive features")
print(f"Error in analysis: {str(e)}")
```

### ✅ **After (Structured Logging)**
```python
logger.info(f"Starting analysis for job {job_id}", extra={"job_id": job_id})
logger.info(f"Found {len(sensitive_features)} sensitive features", 
           extra={"job_id": job_id, "sensitive_features_count": len(sensitive_features)})
logger.error(f"Error in analysis: {str(e)}", exc_info=True, extra={"job_id": job_id})
```

## Benefits

### 🔍 **For Development**
- **Cleaner Console Output**: Only essential messages displayed
- **Debug Information**: Detailed logs available when needed
- **Context Awareness**: Job IDs and operation context in every log

### 🏭 **For Production**
- **Log Aggregation**: JSON logs can be sent to ELK stack, Splunk, etc.
- **Monitoring**: Easy to set up alerts on ERROR level logs
- **Performance Tracking**: Built-in execution time monitoring
- **Audit Trail**: Complete operation history with timestamps

### 📊 **For Operations**
- **Troubleshooting**: Stack traces with full context
- **Metrics**: Automatic counting of bias metrics, processing times
- **Scalability**: Log rotation prevents disk space issues

## Testing

Run the logging test suite:
```bash
python test_logging.py
```

This will:
- Test all logger modules
- Verify structured logging works
- Check execution time tracking
- Test error logging with exceptions
- Generate sample log files

## Integration

The logging system is already integrated into:
- ✅ **Main Application** (`main.py`)
- ✅ **Analysis Service** (`analysis_service.py`)
- ✅ **Upload Service** (`upload_service.py`)
- ✅ **Mitigation Service** (`mitigation_service.py`)
- ✅ **AI Recommendation Service** (`ai_recommendation_service.py`)

## Next Steps

1. **Configure Log Aggregation**: Set up ELK stack or similar for log analysis
2. **Set Up Monitoring**: Configure alerts for ERROR level logs
3. **Add Business Metrics**: Track bias detection rates, processing times
4. **Security**: Ensure no sensitive data appears in logs

## Example Log Output

### Console Output (Clean)
```
2025-09-04 15:36:43 - fairness.analysis - INFO - Analysis started
2025-09-04 15:36:44 - fairness.analysis - INFO - Completed bias detection analysis in 1.00s
```

### Structured Log (Detailed)
```json
{"timestamp": "2025-09-04T15:36:43.627488", "level": "INFO", "module": "analysis_service", "function": "_run_analysis", "line": 750, "message": "Analysis started", "job_id": "abc-123"}
{"timestamp": "2025-09-04T15:36:44.633575", "level": "INFO", "module": "analysis_service", "function": "_detect_bias", "line": 950, "message": "Completed bias detection analysis in 1.00s", "job_id": "abc-123", "execution_time": 1.004738, "bias_metrics_count": 42}
```

This logging system provides enterprise-grade observability while maintaining clean, readable code.
