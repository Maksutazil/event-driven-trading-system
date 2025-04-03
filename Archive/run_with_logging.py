#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script runs the trading system with explicit file logging.
"""

import sys
import os
import logging
import subprocess
import argparse
from datetime import datetime

def setup_logging(log_level):
    """Set up logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"trading_log_{timestamp}.log"
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Console handler with user-specified level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler with DEBUG level (capture everything)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add both handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return log_file

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the trading system with file logging")
    parser.add_argument("--websocket-url", default="wss://pumpportal.fun/api/data", 
                        help="WebSocket URL")
    parser.add_argument("--log-level", default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.log_level)
    print(f"Logging to {log_file}")
    
    # Build command to run the trading system
    cmd = [
        sys.executable,
        "examples/run_sol_trading.py",
        f"--websocket-url={args.websocket_url}",
        f"--log-level={args.log_level}"
    ]
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Terminating process...")
        process.terminate()
        process.wait()
    
    print(f"Process completed with return code {process.returncode}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main() 