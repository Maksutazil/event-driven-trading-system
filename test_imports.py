#!/usr/bin/env python

"""
Test script to check if we can import all necessary modules.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

try:
    print("Importing core modules...")
    from src.core.events import EventBus
    print("✓ Imported EventBus")
    
    from src.core.data import DataFeedManager
    print("✓ Imported DataFeedManager")
    
    from src.core.db import DatabaseClient
    print("✓ Imported DatabaseClient")
    
    from src.core.db import PostgresDataManager
    print("✓ Imported PostgresDataManager")
    
    print("\nAll imports successful!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    
    # Add more debug info
    print("\nCurrent sys.path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Check if the files exist
    print("\nChecking for module files:")
    
    modules_to_check = [
        "src/core/events/__init__.py",
        "src/core/data/__init__.py", 
        "src/core/db/__init__.py",
        "src/core/db/postgres_data_manager.py"
    ]
    
    for module_path in modules_to_check:
        file_path = Path(module_path)
        if file_path.exists():
            print(f"  ✓ {module_path} exists")
        else:
            print(f"  ❌ {module_path} does not exist")
            
except Exception as e:
    print(f"\n❌ Unexpected error: {e}") 