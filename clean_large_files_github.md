# Dealing with Large Files in Git and GitHub

This guide provides detailed instructions for handling large files that were accidentally committed to a Git repository.

## Problem

GitHub has a file size limit of 100MB. When trying to push files larger than this limit, you'll encounter an error. Even if you've added these files to `.gitignore` after they were already committed, they remain in the Git history and will cause problems when pushing.

## Solutions

There are several ways to remove large files from your Git history. Choose the approach that works best for your situation.

### Option 1: Using git-filter-repo (Recommended)

[git-filter-repo](https://github.com/newren/git-filter-repo) is a powerful tool specifically designed for rewriting Git history and is recommended by GitHub.

1. **Install git-filter-repo**

   ```bash
   pip install git-filter-repo
   ```

2. **Create a backup of your repository**

   ```bash
   git clone --mirror https://github.com/yourusername/your-repository.git repo-backup.git
   ```

3. **Run the cleanup Python script**

   ```bash
   python clean_large_files_with_filter_repo.py
   ```

   Or manually run git-filter-repo:

   ```bash
   # Remove a specific directory
   git filter-repo --path trading_venv --invert-paths --force
   
   # Force cleanup
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

4. **Push the changes to GitHub**

   ```bash
   git push origin --force --all
   ```

### Option 2: Using BFG Repo-Cleaner

[BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) is faster and simpler than git-filter-branch.

1. **Download BFG Jar file** from https://rtyley.github.io/bfg-repo-cleaner/

2. **Create a bare clone of your repository**

   ```bash
   git clone --mirror https://github.com/yourusername/your-repository.git repo-mirror.git
   ```

3. **Run BFG to remove the large directory**

   ```bash
   java -jar bfg.jar --delete-folders trading_venv repo-mirror.git
   ```

4. **Clean up and update the repository**

   ```bash
   cd repo-mirror.git
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   git push --force
   ```

### Option 3: Using git-filter-branch

This is the traditional approach but is slower and more complex.

1. **Run the shell script**

   ```bash
   ./clean_large_files.sh
   ```

   Or manually:

   ```bash
   git filter-branch --force --index-filter \
     "git rm -r --cached --ignore-unmatch trading_venv" \
     --prune-empty --tag-name-filter cat -- --all
   ```

2. **Force push to GitHub**

   ```bash
   git push origin --force --all
   ```

## Preventing Future Issues

### Using .gitignore

Make sure your `.gitignore` file includes entries for common large files:

```
# Virtual environments
venv/
trading_venv/
env/
.venv

# Large data files
*.csv
*.h5
data/
```

### Using Git LFS

For repositories that need to track large files, consider using [Git Large File Storage (LFS)](https://git-lfs.github.com/).

1. **Install Git LFS**

   ```bash
   git lfs install
   ```

2. **Track large file types**

   ```bash
   git lfs track "*.csv"
   git lfs track "*.h5"
   ```

3. **Commit the .gitattributes file**

   ```bash
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   ```

## Important Notes

1. **Rewriting history is destructive**: These operations permanently change your Git history.
2. **Force pushing is required**: You'll need to use `--force` when pushing.
3. **Team coordination**: If others are working with the repository, they'll need to re-clone or take special steps to synchronize their local repositories.
4. **Backup first**: Always create a backup before attempting to rewrite history.

## Additional Resources

- [GitHub Documentation: Working with large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git Documentation: git-filter-branch](https://git-scm.com/docs/git-filter-branch)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub Documentation: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)