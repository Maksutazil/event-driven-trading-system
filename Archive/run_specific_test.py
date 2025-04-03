#!/usr/bin/env python3
"""
Script to run a specific integration test with detailed output.

Usage:
    python run_specific_test.py <test_module_path>
    
Example:
    python run_specific_test.py tests.integration.ml_trading.test_model_prediction_flow
"""

import os
import sys
import unittest
import time
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("IntegrationTestRunner")

@contextmanager
def timer(description):
    """Simple context manager to time execution."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description}: {elapsed:.2f} seconds")

def run_specific_test(test_module_path):
    """Run a specific test module and report the results.
    
    Args:
        test_module_path: Dot notation path to the test module
            (e.g., tests.integration.ml_trading.test_model_prediction_flow)
    """
    logger.info(f"Running test module: {test_module_path}")
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    try:
        # Load the test module
        tests = loader.loadTestsFromName(test_module_path)
        
        # Create a test runner that will output verbose results
        runner = unittest.TextTestRunner(verbosity=2)
        
        # Run the tests with timing
        with timer("Test execution completed in"):
            result = runner.run(tests)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        logger.info(f"Tests failed: {len(result.failures)}")
        logger.info(f"Tests with errors: {len(result.errors)}")
        
        # Print failures
        if result.failures:
            logger.error("\n" + "="*80)
            logger.error("FAILURES")
            logger.error("="*80)
            for i, (test, traceback) in enumerate(result.failures, 1):
                logger.error(f"Failure {i}: {test}")
                logger.error(f"{traceback}\n")
        
        # Print errors
        if result.errors:
            logger.error("\n" + "="*80)
            logger.error("ERRORS")
            logger.error("="*80)
            for i, (test, traceback) in enumerate(result.errors, 1):
                logger.error(f"Error {i}: {test}")
                logger.error(f"{traceback}\n")
        
        # Return True if all tests passed, False otherwise
        return len(result.failures) == 0 and len(result.errors) == 0
    
    except Exception as e:
        logger.error(f"Error running test module: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Please provide a test module path.")
        logger.info("Example usage: python run_specific_test.py tests.integration.ml_trading.test_model_prediction_flow")
        sys.exit(1)
    
    test_module_path = sys.argv[1]
    success = run_specific_test(test_module_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 