#!/usr/bin/env python3
"""
Git Large Files Cleanup Script

This script uses git-filter-repo to remove large files from Git history.
git-filter-repo is a more advanced and faster alternative to git-filter-branch.

Requirements:
- git-filter-repo (install via: pip install git-filter-repo)
- Git 2.24.0 or newer

Usage:
python clean_large_files_with_filter_repo.py
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime


def check_requirements():
    """Check if git and git-filter-repo are installed."""
    # Check if git is installed
    try:
        subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Git is not installed or not in PATH")
        return False
    
    # Check if git-filter-repo is installed
    try:
        subprocess.run(['git-filter-repo', '--version'], check=True, stdout=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: git-filter-repo is not installed or not in PATH")
        print("Install it using: pip install git-filter-repo")
        return False


def create_backup_branch():
    """Create a backup branch before making changes."""
    backup_branch = f"backup-before-cleanup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    subprocess.run(['git', 'branch', backup_branch], check=True)
    return backup_branch


def remove_large_files():
    """Remove large files from Git history using git-filter-repo."""
    # Create paths_to_remove.txt file with paths to large files/directories
    with open('paths_to_remove.txt', 'w') as f:
        f.write('trading_venv\n')  # Add more paths as needed
    
    # Run git-filter-repo to remove the specified paths
    print("Removing large files from Git history...")
    subprocess.run([
        'git-filter-repo',
        '--force',
        '--paths-from-file', 'paths_to_remove.txt',
        '--invert-paths',
        '--prune-empty', 'always'
    ], check=True)
    
    # Clean up the temporary file
    os.remove('paths_to_remove.txt')


def cleanup_repository():
    """Clean up references and optimize the repository."""
    print("Cleaning up references and optimizing the repository...")
    
    # Remove the refs/original directory that git-filter-repo leaves behind
    refs_original = os.path.join('.git', 'refs', 'original')
    if os.path.exists(refs_original):
        shutil.rmtree(refs_original)
    
    # Expire all reflogs
    subprocess.run(['git', 'reflog', 'expire', '--expire=now', '--all'], check=True)
    
    # Run garbage collection
    subprocess.run(['git', 'gc', '--prune=now', '--aggressive'], check=True)


def main():
    """Main function to orchestrate the cleanup process."""
    print("Starting the cleanup of large files from Git history...")
    print("This process will rewrite Git history and requires force pushing afterward.")
    print("")
    print("Files/directories to be removed from Git history:")
    print(" - trading_venv/")
    print("")
    print("Warning: This will rewrite Git history. Anyone else using this repository")
    print("will need to re-clone it after you force push the changes.")
    print("")
    
    # Confirm with the user
    response = input("Do you want to continue? (y/n) ").lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Check if requirements are met
    if not check_requirements():
        return
    
    # Create a backup branch
    backup_branch = create_backup_branch()
    print(f"Created backup branch: {backup_branch}")
    
    # Remove large files
    remove_large_files()
    
    # Clean up and optimize the repository
    cleanup_repository()
    
    # Final message
    print("\nCleanup completed!")
    print("The large files have been removed from Git history.")
    print("To complete the process, you need to force push these changes to GitHub:")
    print("")
    print("  git push origin --force --all")
    print("")
    print("Warning: This will overwrite the remote history. Make sure all collaborators are aware.")
    print(f"A backup branch '{backup_branch}' has been created locally in case you need to revert changes.")


if __name__ == "__main__":
    main()