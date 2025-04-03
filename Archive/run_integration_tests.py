#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test Runner

This script runs all the integration tests for the event-driven trading system
and provides a summary of the results.
"""

import os
import sys
import unittest
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all integration tests and provide a summary of results."""
    start_time = time.time()
    
    # Print header
    print("\n" + "="*80)
    print(f"RUNNING INTEGRATION TESTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Discover and run tests
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests/integration')
    
    # Make sure the directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        return 1
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    # Create test result collection
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Calculate timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Total time: {duration:.2f} seconds")
    print("="*80)
    
    # Print failures and errors
    if result.failures:
        print("\nFAILURES:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print("-" * 70)
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print("-" * 70)
            print(traceback)
    
    # Return status code (0 for success, 1 for failure)
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_module):
    """Run a specific test module."""
    start_time = time.time()
    
    # Print header
    print("\n" + "="*80)
    print(f"RUNNING TEST MODULE: {test_module} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load and run the specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    # Create test result collection
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Calculate timing
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Total time: {duration:.2f} seconds")
    print("="*80)
    
    # Return status code (0 for success, 1 for failure)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Check if a specific test module was specified
    if len(sys.argv) > 1:
        test_module = sys.argv[1]
        sys.exit(run_specific_test(test_module))
    else:
        # Run all tests
        sys.exit(run_tests()) 