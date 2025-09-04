#!/usr/bin/env python3
"""
Simple test script to verify the logging implementation is working correctly.
Run this to test the logging system without starting the full application.
"""

import sys
import os
import time

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.utils.logger import (
    get_logger, get_analysis_logger, get_upload_logger, 
    get_mitigation_logger, get_api_logger, get_ai_logger,
    LogExecutionTime, log_bias_detection_metrics
)

def test_basic_logging():
    """Test basic logging functionality"""
    print("=" * 50)
    print("TESTING BASIC LOGGING")
    print("=" * 50)
    
    # Test different loggers
    api_logger = get_api_logger()
    analysis_logger = get_analysis_logger()
    upload_logger = get_upload_logger()
    
    # Test different log levels
    api_logger.debug("This is a debug message")
    api_logger.info("This is an info message")
    api_logger.warning("This is a warning message")
    api_logger.error("This is an error message")
    
    # Test structured logging with extra fields
    analysis_logger.info(
        "Analysis started", 
        extra={"job_id": "test-123", "bias_metrics_count": 42}
    )
    
    upload_logger.info("File uploaded successfully", extra={"file_size": 1024})

def test_execution_time_logging():
    """Test execution time logging context manager"""
    print("\n" + "=" * 50)
    print("TESTING EXECUTION TIME LOGGING")
    print("=" * 50)
    
    logger = get_analysis_logger()
    
    # Test successful operation
    with LogExecutionTime(logger, "bias detection analysis", "job-456"):
        time.sleep(1)  # Simulate some work
        print("Simulating bias detection work...")
    
    # Test failed operation
    try:
        with LogExecutionTime(logger, "model validation", "job-789"):
            time.sleep(0.5)
            raise ValueError("Simulated validation error")
    except ValueError:
        print("Expected error caught")

def test_bias_detection_decorator():
    """Test bias detection metrics logging decorator"""
    print("\n" + "=" * 50)
    print("TESTING BIAS DETECTION DECORATOR")
    print("=" * 50)
    
    logger = get_analysis_logger()
    
    @log_bias_detection_metrics(logger)
    def mock_bias_detection():
        """Mock bias detection function"""
        time.sleep(0.5)
        # Return mock bias metrics (list of objects)
        return [
            {"metric": "statistical_parity", "value": 0.15},
            {"metric": "disparate_impact", "value": 0.8},
            {"metric": "equal_opportunity", "value": 0.12}
        ]
    
    try:
        metrics = mock_bias_detection()
        print(f"Bias detection returned {len(metrics)} metrics")
    except Exception as e:
        print(f"Error in bias detection: {e}")

def test_different_modules():
    """Test logging from different modules"""
    print("\n" + "=" * 50)
    print("TESTING DIFFERENT MODULE LOGGERS")
    print("=" * 50)
    
    # Test all module loggers
    modules = {
        "API": get_api_logger(),
        "Analysis": get_analysis_logger(),
        "Upload": get_upload_logger(),
        "Mitigation": get_mitigation_logger(),
        "AI": get_ai_logger()
    }
    
    for module_name, logger in modules.items():
        logger.info(f"Testing {module_name} module logger")
        logger.debug(f"Debug message from {module_name}")
        logger.warning(f"Warning from {module_name} module")

def test_error_logging():
    """Test error logging with exception details"""
    print("\n" + "=" * 50)
    print("TESTING ERROR LOGGING WITH EXCEPTIONS")
    print("=" * 50)
    
    logger = get_analysis_logger()
    
    try:
        # Simulate a complex error
        data = {"key": "value"}
        result = data["missing_key"]
    except KeyError as e:
        logger.error(
            "Failed to process data structure", 
            exc_info=True,
            extra={"data_keys": list(data.keys()), "job_id": "error-test-123"}
        )
    
    try:
        # Simulate another type of error
        numbers = [1, 2, 3]
        value = numbers[10]
    except IndexError as e:
        logger.error(
            "Index out of range error",
            exc_info=True,
            extra={"list_length": len(numbers), "attempted_index": 10}
        )

def main():
    """Run all logging tests"""
    print("🚀 Starting Fairness Assessment Platform - Logging Test Suite")
    print(f"Log files will be created in: {os.path.abspath('logs')}")
    
    try:
        test_basic_logging()
        test_execution_time_logging()
        test_bias_detection_decorator()
        test_different_modules()
        test_error_logging()
        
        print("\n" + "=" * 50)
        print("✅ ALL LOGGING TESTS COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print("\nCheck the 'logs' directory for generated log files:")
        print("- *.log files contain detailed formatted logs")
        print("- *_structured.log files contain JSON formatted logs")
        
    except Exception as e:
        print(f"\n❌ LOGGING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
