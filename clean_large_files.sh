#!/bin/bash

# Script to remove large files from Git history
# This script will help clean up the Git repository by removing large files
# from the entire commit history.

echo "Starting the cleanup of large files from Git history..."
echo "This process will rewrite Git history and requires force pushing afterward."
echo ""
echo "Files/directories to be removed from Git history:"
echo " - trading_venv/"
echo ""
echo "Warning: This will rewrite Git history. Anyone else using this repository"
echo "will need to re-clone it after you force push the changes."
echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Make sure we have the latest version of the repo
echo "Fetching the latest changes..."
git fetch --all

# Create a backup branch
backup_branch="backup-before-cleanup-$(date +%Y%m%d%H%M%S)"
echo "Creating a backup branch: $backup_branch"
git branch $backup_branch

# Install BFG Repo Cleaner if not already available
# Note: BFG is a more efficient alternative to git-filter-branch
echo "Checking for BFG Repo Cleaner..."
if ! command -v bfg &>/dev/null; then
    echo "BFG not found. Please install BFG Repo Cleaner and try again."
    echo "You can download it from: https://rtyley.github.io/bfg-repo-cleaner/"
    echo ""
    echo "Alternative approach: You can use git filter-branch instead."
    echo "Would you like to use git filter-branch instead? (y/n)"
    read -p "" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Operation cancelled. Please install BFG and try again."
        exit 1
    fi
    
    echo "Using git filter-branch instead..."
    echo "Removing trading_venv directory from Git history..."
    git filter-branch --force --index-filter \
      "git rm -r --cached --ignore-unmatch trading_venv" \
      --prune-empty --tag-name-filter cat -- --all
else
    echo "BFG found. Using BFG to remove large files..."
    # Create a mirror of the repository
    echo "Creating a mirror of the repository..."
    git clone --mirror $(git config --get remote.origin.url) repo-mirror.git
    cd repo-mirror.git
    
    # Use BFG to remove the directories
    echo "Removing trading_venv directory from Git history..."
    bfg --delete-folders trading_venv
    
    # Clean up and optimize the repository
    echo "Cleaning up and optimizing the repository..."
    git reflog expire --expire=now --all && git gc --prune=now --aggressive
    
    # Move back to the original repository
    cd ..
    
    # Get the changes back to the original repository
    echo "Applying changes to the original repository..."
    git remote add mirror ./repo-mirror.git
    git fetch mirror --prune
    
    # Reset to the cleaned state
    git reset --hard mirror/main
    
    # Remove the temporary mirror
    git remote remove mirror
    rm -rf repo-mirror.git
fi

# Clean up reflog and optimize the repository
echo "Cleaning up references and optimizing the repository..."
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Cleanup completed!"
echo ""
echo "The large files have been removed from Git history."
echo "To complete the process, you need to force push these changes to GitHub:"
echo ""
echo "  git push origin --force --all"
echo ""
echo "Warning: This will overwrite the remote history. Make sure all collaborators are aware."
echo "A backup branch '$backup_branch' has been created locally in case you need to revert changes."