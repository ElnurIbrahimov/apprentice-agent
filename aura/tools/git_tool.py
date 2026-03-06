"""Git tool for repository management and version control operations.

This tool provides git operations using subprocess calls to the git CLI,
avoiding external dependencies like GitPython.
"""

import subprocess
import os
from typing import Optional
from pathlib import Path


class GitTool:
    """Tool for interacting with Git repositories."""

    def __init__(self):
        """Initialize the git tool."""
        self._git_available = self._check_git_installed()

    def _check_git_installed(self) -> bool:
        """Check if git is available on the system."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _run_git(self, args: list, cwd: str = ".") -> dict:
        """Run a git command and return the result.

        Args:
            args: List of git command arguments (without 'git' prefix)
            cwd: Working directory for the command

        Returns:
            dict with success, output, and error fields
        """
        if not self._git_available:
            return {
                "success": False,
                "error": "Git is not installed or not in PATH"
            }

        # Resolve the path
        repo_path = Path(cwd).resolve()
        if not repo_path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {cwd}"
            }

        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout.strip(),
                    "stderr": result.stderr.strip() if result.stderr else None
                }
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                return self._parse_git_error(error_msg, args)

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Git command timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run git command: {str(e)}"
            }

    def _parse_git_error(self, error_msg: str, args: list) -> dict:
        """Parse git error messages and provide helpful responses."""
        error_lower = error_msg.lower()

        # Not a git repository
        if "not a git repository" in error_lower:
            return {
                "success": False,
                "error": "Not a git repository. Run 'git init' or navigate to a git repository.",
                "error_type": "not_a_repo"
            }

        # Authentication failures
        if any(x in error_lower for x in ["authentication failed", "could not read username", "permission denied"]):
            return {
                "success": False,
                "error": "Authentication failed. Check your credentials or SSH keys.",
                "error_type": "auth_failure"
            }

        # Merge conflicts
        if any(x in error_lower for x in ["merge conflict", "conflict", "unmerged files"]):
            return {
                "success": False,
                "error": "Merge conflict detected. Resolve conflicts manually before continuing.",
                "error_type": "merge_conflict"
            }

        # Remote not found
        if "repository not found" in error_lower or "could not resolve host" in error_lower:
            return {
                "success": False,
                "error": "Remote repository not found. Check the URL and your network connection.",
                "error_type": "remote_not_found"
            }

        # Nothing to commit
        if "nothing to commit" in error_lower:
            return {
                "success": True,
                "output": "Nothing to commit, working tree clean",
                "warning": "No changes to commit"
            }

        # Branch not found
        if "did not match any" in error_lower or "unknown revision" in error_lower:
            return {
                "success": False,
                "error": f"Branch or reference not found: {error_msg}",
                "error_type": "ref_not_found"
            }

        # Already up to date
        if "already up to date" in error_lower:
            return {
                "success": True,
                "output": "Already up to date",
                "warning": "No new changes to pull"
            }

        # Default error
        return {
            "success": False,
            "error": error_msg,
            "command": f"git {' '.join(args)}"
        }

    def status(self, repo_path: str = ".") -> dict:
        """Get the current repository status.

        Shows the current branch, staged changes, unstaged changes,
        and untracked files.

        Args:
            repo_path: Path to the git repository (default: current directory)

        Returns:
            dict with:
                - success: bool
                - output: Formatted status string for LLM to display verbatim
                - branch: Current branch name
                - clean: bool indicating if working tree is clean

        Example:
            >>> tool = GitTool()
            >>> result = tool.status(".")
            >>> result['branch']
            'main'
        """
        # Get current branch
        branch_result = self._run_git(["branch", "--show-current"], repo_path)
        if not branch_result.get("success") and "not_a_repo" in str(branch_result.get("error_type", "")):
            return branch_result

        branch = branch_result.get("output", "unknown")

        # Get status in porcelain format for easier parsing
        status_result = self._run_git(["status", "--porcelain"], repo_path)
        if not status_result.get("success"):
            return status_result

        staged = []
        unstaged = []
        untracked = []

        for line in status_result.get("output", "").split("\n"):
            if not line:
                continue

            # Porcelain format: XY filename
            # X = index status, Y = work tree status
            if len(line) >= 3:
                index_status = line[0]
                worktree_status = line[1]
                filename = line[3:]

                # Staged changes (added, modified, deleted, renamed in index)
                if index_status in "AMDRC":
                    staged.append(f"  {self._status_code(index_status)}: {filename}")

                # Unstaged changes (modified, deleted in work tree)
                if worktree_status in "MD":
                    unstaged.append(f"  {self._status_code(worktree_status)}: {filename}")

                # Untracked files
                if index_status == "?" and worktree_status == "?":
                    untracked.append(f"  {filename}")

        # Check if ahead/behind remote
        ahead_behind = self._get_ahead_behind(repo_path)

        # Build formatted output string
        lines = [f"ACTUAL GIT STATUS:"]
        lines.append(f"Branch: {branch}")

        if ahead_behind.get("ahead", 0) > 0:
            lines.append(f"Ahead of origin by {ahead_behind['ahead']} commit(s)")
        if ahead_behind.get("behind", 0) > 0:
            lines.append(f"Behind origin by {ahead_behind['behind']} commit(s)")

        if staged:
            lines.append(f"\nStaged for commit ({len(staged)}):")
            lines.extend(staged)

        if unstaged:
            lines.append(f"\nUnstaged changes ({len(unstaged)}):")
            lines.extend(unstaged)

        if untracked:
            lines.append(f"\nUntracked files ({len(untracked)}):")
            lines.extend(untracked)

        if not staged and not unstaged and not untracked:
            lines.append("\nWorking tree clean - nothing to commit")

        return {
            "success": True,
            "output": "\n".join(lines),
            "branch": branch,
            "clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
            "staged_count": len(staged),
            "unstaged_count": len(unstaged),
            "untracked_count": len(untracked),
            **ahead_behind
        }

    def _status_code(self, code: str) -> str:
        """Convert git status code to human-readable string."""
        codes = {
            "A": "added",
            "M": "modified",
            "D": "deleted",
            "R": "renamed",
            "C": "copied",
            "?": "untracked"
        }
        return codes.get(code, code)

    def _get_ahead_behind(self, repo_path: str) -> dict:
        """Get ahead/behind count relative to upstream."""
        result = self._run_git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"], repo_path)
        if result.get("success"):
            parts = result.get("output", "0\t0").split()
            if len(parts) >= 2:
                return {
                    "behind": int(parts[0]),
                    "ahead": int(parts[1])
                }
        return {"behind": 0, "ahead": 0}

    def log(self, repo_path: str = ".", count: int = 5) -> dict:
        """Get recent commit history.

        Args:
            repo_path: Path to the git repository
            count: Number of commits to retrieve (default: 5)

        Returns:
            dict with:
                - success: bool
                - output: Formatted log string for LLM to display verbatim
                - count: Number of commits returned

        Example:
            >>> tool = GitTool()
            >>> result = tool.log(".", count=3)
            >>> print(result['output'])
        """
        count = min(max(1, count), 100)  # Clamp between 1-100

        # Get formatted log output
        result = self._run_git(
            ["log", f"-{count}", "--pretty=format:%h %s (%cr) - %an"],
            repo_path
        )

        if not result.get("success"):
            return result

        log_output = result.get("output", "").strip()
        if not log_output:
            return {
                "success": True,
                "output": "ACTUAL GIT LOG:\nNo commits found.",
                "count": 0
            }

        commit_count = len(log_output.split("\n"))
        return {
            "success": True,
            "output": f"ACTUAL GIT LOG ({commit_count} commits):\n{log_output}",
            "count": commit_count
        }

    def diff(self, repo_path: str = ".", file: Optional[str] = None) -> dict:
        """Show changes in the repository.

        Args:
            repo_path: Path to the git repository
            file: Specific file to diff (default: all files)

        Returns:
            dict with:
                - success: bool
                - output: Formatted diff string for LLM to display verbatim
                - has_changes: bool

        Example:
            >>> tool = GitTool()
            >>> result = tool.diff(".", file="README.md")
            >>> print(result['output'])
        """
        # Get diff stat summary
        stat_args = ["diff", "--stat"]
        if file:
            stat_args.extend(["--", file])
        stat_result = self._run_git(stat_args, repo_path)

        if not stat_result.get("success"):
            return stat_result

        stat_output = stat_result.get("output", "").strip()

        if not stat_output:
            return {
                "success": True,
                "output": "ACTUAL GIT DIFF:\nNo changes detected.",
                "has_changes": False
            }

        # Get the actual diff content (limited to avoid huge output)
        diff_args = ["diff", "--no-color"]
        if file:
            diff_args.extend(["--", file])
        diff_result = self._run_git(diff_args, repo_path)

        diff_content = diff_result.get("output", "").strip() if diff_result.get("success") else ""

        # Truncate if too long
        max_diff_lines = 50
        diff_lines = diff_content.split("\n")
        if len(diff_lines) > max_diff_lines:
            diff_content = "\n".join(diff_lines[:max_diff_lines]) + f"\n... ({len(diff_lines) - max_diff_lines} more lines)"

        output_lines = ["ACTUAL GIT DIFF:"]
        output_lines.append(f"Summary:\n{stat_output}")
        if diff_content:
            output_lines.append(f"\nChanges:\n{diff_content}")

        return {
            "success": True,
            "output": "\n".join(output_lines),
            "has_changes": True
        }

    def branch(self, repo_path: str = ".") -> dict:
        """List all branches and show the current one.

        Args:
            repo_path: Path to the git repository

        Returns:
            dict with:
                - success: bool
                - output: Formatted branch list for LLM to display verbatim
                - current: Current branch name

        Example:
            >>> tool = GitTool()
            >>> result = tool.branch(".")
            >>> print(result['output'])
        """
        # Get all branches with verbose info
        result = self._run_git(["branch", "-vv"], repo_path)
        if not result.get("success"):
            return result

        branch_output = result.get("output", "").strip()

        # Find current branch
        current = None
        for line in branch_output.split("\n"):
            if line.startswith("*"):
                current = line.split()[1] if len(line.split()) > 1 else None
                break

        # Get remote branches too
        remote_result = self._run_git(["branch", "-r"], repo_path)
        remote_output = remote_result.get("output", "").strip() if remote_result.get("success") else ""

        output_lines = ["ACTUAL GIT BRANCHES:"]
        output_lines.append(f"Current branch: {current or 'unknown'}")
        output_lines.append(f"\nLocal branches:\n{branch_output}")

        if remote_output:
            output_lines.append(f"\nRemote branches:\n{remote_output}")

        return {
            "success": True,
            "output": "\n".join(output_lines),
            "current": current
        }

    def add(self, repo_path: str = ".", files: str = ".") -> dict:
        """Stage files for commit.

        Args:
            repo_path: Path to the git repository
            files: Files to stage (default: "." for all)
                   Can be a single file, pattern, or space-separated list

        Returns:
            dict with:
                - success: bool
                - message: Description of what was staged

        Example:
            >>> tool = GitTool()
            >>> result = tool.add(".", files="README.md")
            >>> result['success']
            True
        """
        # Handle multiple files
        if isinstance(files, str):
            file_list = files.split() if " " in files else [files]
        else:
            file_list = list(files)

        result = self._run_git(["add"] + file_list, repo_path)
        if not result.get("success"):
            return result

        # Get what was actually staged
        status = self.status(repo_path)
        staged_count = status.get("staged_count", 0) if status.get("success") else 0

        return {
            "success": True,
            "message": f"Staged {staged_count} file(s)",
            "files": file_list,
            "staged_count": staged_count
        }

    def commit(self, repo_path: str = ".", message: str = "") -> dict:
        """Create a commit with the staged changes.

        Args:
            repo_path: Path to the git repository
            message: Commit message (required)

        Returns:
            dict with:
                - success: bool
                - hash: Commit hash
                - message: Commit message

        Example:
            >>> tool = GitTool()
            >>> result = tool.commit(".", message="Fix bug in login")
            >>> result['hash']
            'abc1234'
        """
        if not message:
            return {
                "success": False,
                "error": "Commit message is required"
            }

        result = self._run_git(["commit", "-m", message], repo_path)
        if not result.get("success"):
            return result

        # Get the commit hash
        hash_result = self._run_git(["rev-parse", "HEAD"], repo_path)
        commit_hash = hash_result.get("output", "")[:7] if hash_result.get("success") else "unknown"

        return {
            "success": True,
            "hash": commit_hash,
            "message": message,
            "output": result.get("output", "")
        }

    def push(self, repo_path: str = ".", remote: str = "origin", branch: Optional[str] = None) -> dict:
        """Push commits to a remote repository.

        Args:
            repo_path: Path to the git repository
            remote: Remote name (default: "origin")
            branch: Branch to push (default: current branch)

        Returns:
            dict with:
                - success: bool
                - message: Push result message

        Example:
            >>> tool = GitTool()
            >>> result = tool.push(".", remote="origin", branch="main")
            >>> result['success']
            True
        """
        args = ["push", remote]
        if branch:
            args.append(branch)

        result = self._run_git(args, repo_path)
        if not result.get("success"):
            return result

        output = result.get("output", "") or result.get("stderr", "")
        return {
            "success": True,
            "message": f"Pushed to {remote}" + (f"/{branch}" if branch else ""),
            "output": output,
            "remote": remote,
            "branch": branch
        }

    def pull(self, repo_path: str = ".", remote: str = "origin", branch: Optional[str] = None) -> dict:
        """Pull changes from a remote repository.

        Args:
            repo_path: Path to the git repository
            remote: Remote name (default: "origin")
            branch: Branch to pull (default: current branch)

        Returns:
            dict with:
                - success: bool
                - message: Pull result message
                - updated: Whether any changes were pulled

        Example:
            >>> tool = GitTool()
            >>> result = tool.pull(".", remote="origin")
            >>> result['updated']
            True
        """
        args = ["pull", remote]
        if branch:
            args.append(branch)

        result = self._run_git(args, repo_path)
        if not result.get("success"):
            return result

        output = result.get("output", "")
        updated = "already up to date" not in output.lower()

        return {
            "success": True,
            "message": f"Pulled from {remote}" + (f"/{branch}" if branch else ""),
            "output": output,
            "updated": updated,
            "remote": remote,
            "branch": branch
        }

    def clone(self, url: str, destination: Optional[str] = None) -> dict:
        """Clone a repository from a URL.

        Args:
            url: Repository URL to clone
            destination: Local directory name (default: derived from URL)

        Returns:
            dict with:
                - success: bool
                - path: Path to the cloned repository
                - message: Clone result message

        Example:
            >>> tool = GitTool()
            >>> result = tool.clone("https://github.com/user/repo.git")
            >>> result['path']
            'repo'
        """
        if not url:
            return {
                "success": False,
                "error": "Repository URL is required"
            }

        args = ["clone", url]
        if destination:
            args.append(destination)

        # Clone runs in current directory, not in a repo
        result = self._run_git(args, ".")
        if not result.get("success"):
            return result

        # Determine the destination path
        if destination:
            cloned_path = destination
        else:
            # Extract from URL: https://github.com/user/repo.git -> repo
            import re
            match = re.search(r'/([^/]+?)(?:\.git)?$', url)
            cloned_path = match.group(1) if match else "repository"

        return {
            "success": True,
            "path": cloned_path,
            "url": url,
            "message": f"Cloned {url} to {cloned_path}",
            "output": result.get("output", "") or result.get("stderr", "")
        }

    def stash(self, repo_path: str = ".", action: str = "push", message: Optional[str] = None) -> dict:
        """Manage stashed changes.

        Args:
            repo_path: Path to the git repository
            action: Stash action - "push", "pop", "list", "drop", "clear"
            message: Message for stash push (optional)

        Returns:
            dict with:
                - success: bool
                - message: Result message
                - stashes: List of stashes (for list action)

        Example:
            >>> tool = GitTool()
            >>> result = tool.stash(".", action="push", message="WIP: feature")
            >>> result['success']
            True
        """
        action = action.lower()

        if action == "push":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
        elif action == "pop":
            args = ["stash", "pop"]
        elif action == "list":
            args = ["stash", "list"]
        elif action == "drop":
            args = ["stash", "drop"]
        elif action == "clear":
            args = ["stash", "clear"]
        else:
            return {
                "success": False,
                "error": f"Unknown stash action: {action}. Use: push, pop, list, drop, clear"
            }

        result = self._run_git(args, repo_path)
        if not result.get("success"):
            return result

        response = {
            "success": True,
            "action": action,
            "message": f"Stash {action} completed",
            "output": result.get("output", "")
        }

        # Parse stash list
        if action == "list":
            stashes = []
            for line in result.get("output", "").split("\n"):
                if line:
                    # Format: stash@{0}: WIP on main: abc1234 message
                    import re
                    match = re.match(r'(stash@\{\d+\}):\s*(.+)', line)
                    if match:
                        stashes.append({
                            "ref": match.group(1),
                            "description": match.group(2)
                        })
            response["stashes"] = stashes
            response["count"] = len(stashes)

        return response

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a git action from natural language.

        Args:
            action: Action to perform
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # Extract repo path if mentioned
        repo_path = kwargs.get("repo_path", ".")
        import re
        path_match = re.search(r'(?:in|at|for)\s+["\']?([^"\']+)["\']?', action)
        if path_match:
            potential_path = path_match.group(1).strip()
            if os.path.isdir(potential_path):
                repo_path = potential_path

        # Status
        if any(x in action_lower for x in ["status", "what changed", "current state"]):
            return self.status(repo_path)

        # Log
        if any(x in action_lower for x in ["log", "history", "recent commits", "commits"]):
            count = 5
            count_match = re.search(r'(\d+)\s*(?:commits?|entries)', action_lower)
            if count_match:
                count = int(count_match.group(1))
            return self.log(repo_path, count)

        # Diff
        if any(x in action_lower for x in ["diff", "changes", "what's different"]):
            file = None
            file_match = re.search(r'(?:for|of|in)\s+["\']?([^\s"\']+\.\w+)["\']?', action)
            if file_match:
                file = file_match.group(1)
            return self.diff(repo_path, file)

        # Branch
        if any(x in action_lower for x in ["branch", "branches"]) and not any(x in action_lower for x in ["push", "pull"]):
            return self.branch(repo_path)

        # Add/Stage
        if any(x in action_lower for x in ["add", "stage"]):
            files = "."
            file_match = re.search(r'(?:add|stage)\s+["\']?([^"\']+)["\']?', action_lower)
            if file_match:
                files = file_match.group(1).strip()
            return self.add(repo_path, files)

        # Commit
        if "commit" in action_lower:
            message = kwargs.get("message", "")
            if not message:
                msg_match = re.search(r'(?:message|msg|with)\s*[:\s]*["\']([^"\']+)["\']', action)
                if msg_match:
                    message = msg_match.group(1)
                else:
                    # Try to extract anything in quotes
                    quote_match = re.search(r'["\']([^"\']+)["\']', action)
                    if quote_match:
                        message = quote_match.group(1)
            return self.commit(repo_path, message)

        # Push
        if "push" in action_lower:
            remote = kwargs.get("remote", "origin")
            branch = kwargs.get("branch")
            return self.push(repo_path, remote, branch)

        # Pull
        if "pull" in action_lower:
            remote = kwargs.get("remote", "origin")
            branch = kwargs.get("branch")
            return self.pull(repo_path, remote, branch)

        # Clone
        if "clone" in action_lower:
            url_match = re.search(r'(https?://[^\s]+|git@[^\s]+)', action)
            if url_match:
                url = url_match.group(1)
                dest = kwargs.get("destination")
                return self.clone(url, dest)
            return {"success": False, "error": "No repository URL found"}

        # Stash
        if "stash" in action_lower:
            if "pop" in action_lower:
                return self.stash(repo_path, "pop")
            elif "list" in action_lower:
                return self.stash(repo_path, "list")
            elif "drop" in action_lower:
                return self.stash(repo_path, "drop")
            elif "clear" in action_lower:
                return self.stash(repo_path, "clear")
            else:
                message = None
                msg_match = re.search(r'["\']([^"\']+)["\']', action)
                if msg_match:
                    message = msg_match.group(1)
                return self.stash(repo_path, "push", message)

        return {
            "success": False,
            "error": f"Unknown git action: {action}",
            "hint": "Try: status, log, diff, branch, add, commit, push, pull, clone, stash"
        }


# Singleton instance for easy import
git_tool = GitTool()
