import os
import time
import glob
import sys

def get_latest_log_file():
    """Get the most recent log file in the current directory."""
    log_files = glob.glob("*.log")
    if not log_files:
        return None
    # Return the most recently modified file
    return max(log_files, key=os.path.getmtime)

def tail_log(file_path, n=50):
    """Show the last n lines of a log file."""
    if not file_path or not os.path.exists(file_path):
        print(f"Log file not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            last_n = lines[-n:] if len(lines) > n else lines
            print(f"\nShowing last {len(last_n)} lines from {file_path}:\n")
            for line in last_n:
                print(line.strip())
    except Exception as e:
        print(f"Error reading log file: {e}")

def main():
    # Check if a specific file was provided
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = get_latest_log_file()
        if not log_file:
            print("No log files found in the current directory.")
            return
    
    # Number of lines to show
    n = 50
    if len(sys.argv) > 2:
        try:
            n = int(sys.argv[2])
        except ValueError:
            pass
    
    tail_log(log_file, n)

if __name__ == "__main__":
    main() 