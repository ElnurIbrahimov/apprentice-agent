"""GitHub Awareness Tool — understand your repositories, PRs, issues and activity.

Uses the GitHub REST API to give AURA awareness of your development activity.
Configure with GITHUB_TOKEN env var for authenticated access (higher rate limits).

Features:
- Weekly activity summaries across repos
- List open PRs and issues
- Recent commit history
- Repository stats
- Code search
- Notifications
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_TIMEOUT = 15


class GitHubTool:
    """GitHub API awareness — commits, PRs, issues, repo stats for your projects."""

    name = "github"
    description = "GitHub awareness — weekly dev summaries, open PRs/issues, recent commits, repo stats"

    def __init__(self):
        self._token = os.getenv("GITHUB_TOKEN", "")
        self._username = os.getenv("GITHUB_USERNAME", "")
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            self._headers["Authorization"] = f"Bearer {self._token}"
            logger.info("[GitHub] Authenticated with token")
        else:
            logger.info("[GitHub] No GITHUB_TOKEN — using unauthenticated (60 req/hr limit)")

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: Optional[Dict] = None) -> tuple[Optional[Dict | List], Optional[str]]:
        """GET from GitHub API. Returns (data, error)."""
        if not REQUESTS_AVAILABLE:
            return None, "requests library not available"
        try:
            url = f"{GITHUB_API}{path}" if path.startswith("/") else path
            resp = requests.get(url, headers=self._headers, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 401:
                return None, "GitHub authentication failed — check GITHUB_TOKEN"
            if resp.status_code == 403:
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                return None, f"Rate limited (remaining: {remaining}). Set GITHUB_TOKEN for higher limits."
            if resp.status_code == 404:
                return None, f"Not found: {path}"
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    def _parse_repo(self, repo_str: str) -> tuple[str, str]:
        """Parse 'owner/repo' string."""
        if "/" in repo_str:
            owner, repo = repo_str.split("/", 1)
            return owner, repo
        return self._username, repo_str

    def _since_days(self, days: int) -> str:
        dt = datetime.now(timezone.utc) - timedelta(days=days)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def weekly_summary(self, repo: Optional[str] = None) -> Dict:
        """Generate a weekly activity summary.

        Args:
            repo: 'owner/repo' or just 'repo' (uses GITHUB_USERNAME as owner).
                  If None, summarizes across all user repos.
        """
        since = self._since_days(7)
        if repo:
            return self._repo_weekly_summary(repo, since)
        return self._user_weekly_summary(since)

    def _repo_weekly_summary(self, repo: str, since: str) -> Dict:
        owner, repo_name = self._parse_repo(repo)
        summary = {"repo": f"{owner}/{repo_name}", "period": "last 7 days"}

        # Commits
        commits_data, err = self._get(f"/repos/{owner}/{repo_name}/commits", {"since": since, "per_page": 100})
        if err:
            return {"success": False, "error": err}
        commits = commits_data or []

        # PRs
        prs_data, _ = self._get(f"/repos/{owner}/{repo_name}/pulls", {"state": "all", "per_page": 30})
        recent_prs = [
            pr for pr in (prs_data or [])
            if pr.get("created_at", "") >= since or pr.get("updated_at", "") >= since
        ]

        # Issues
        issues_data, _ = self._get(f"/repos/{owner}/{repo_name}/issues", {"state": "all", "since": since, "per_page": 30})
        issues = [i for i in (issues_data or []) if "pull_request" not in i]

        # Authors
        author_counts: Dict[str, int] = {}
        for c in commits:
            author = c.get("commit", {}).get("author", {}).get("name", "unknown")
            author_counts[author] = author_counts.get(author, 0) + 1

        summary.update({
            "success": True,
            "commits": {
                "count": len(commits),
                "by_author": author_counts,
                "recent": [
                    {
                        "sha": c["sha"][:7],
                        "message": c["commit"]["message"].split("\n")[0][:80],
                        "author": c["commit"]["author"]["name"],
                        "date": c["commit"]["author"]["date"][:10],
                    }
                    for c in commits[:10]
                ],
            },
            "pull_requests": {
                "open": len([p for p in recent_prs if p["state"] == "open"]),
                "merged": len([p for p in recent_prs if p.get("merged_at")]),
                "items": [
                    {
                        "number": p["number"],
                        "title": p["title"],
                        "state": "merged" if p.get("merged_at") else p["state"],
                        "author": p["user"]["login"],
                    }
                    for p in recent_prs[:10]
                ],
            },
            "issues": {
                "opened": len([i for i in issues if i.get("created_at", "") >= since]),
                "closed": len([i for i in issues if i.get("closed_at", "") and i.get("closed_at", "") >= since]),
                "items": [
                    {"number": i["number"], "title": i["title"], "state": i["state"]}
                    for i in issues[:5]
                ],
            },
        })
        return summary

    def _user_weekly_summary(self, since: str) -> Dict:
        if not self._username:
            return {"success": False, "error": "Set GITHUB_USERNAME in .env for user-wide summaries"}

        repos_data, err = self._get(f"/users/{self._username}/repos", {"sort": "pushed", "per_page": 10})
        if err:
            # Try authenticated user endpoint
            repos_data, err = self._get("/user/repos", {"sort": "pushed", "per_page": 10})
        if err:
            return {"success": False, "error": err}

        active_repos = []
        total_commits = 0
        for repo in (repos_data or []):
            if repo.get("pushed_at", "") < since[:10]:
                continue
            r_name = repo["full_name"]
            commits, _ = self._get(f"/repos/{r_name}/commits", {"since": since, "per_page": 10})
            count = len(commits or [])
            if count > 0:
                active_repos.append({"repo": r_name, "commits": count})
                total_commits += count

        return {
            "success": True,
            "period": "last 7 days",
            "username": self._username,
            "total_commits": total_commits,
            "active_repos": sorted(active_repos, key=lambda x: x["commits"], reverse=True),
        }

    def list_prs(self, repo: str, state: str = "open") -> Dict:
        """List pull requests for a repository.

        Args:
            repo: 'owner/repo' string
            state: 'open', 'closed', or 'all'
        """
        owner, repo_name = self._parse_repo(repo)
        data, err = self._get(f"/repos/{owner}/{repo_name}/pulls", {"state": state, "per_page": 30})
        if err:
            return {"success": False, "error": err}
        prs = data or []
        return {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "state": state,
            "count": len(prs),
            "pull_requests": [
                {
                    "number": pr["number"],
                    "title": pr["title"],
                    "author": pr["user"]["login"],
                    "state": pr["state"],
                    "created_at": pr["created_at"][:10],
                    "updated_at": pr["updated_at"][:10],
                    "labels": [l["name"] for l in pr.get("labels", [])],
                    "url": pr["html_url"],
                }
                for pr in prs
            ],
        }

    def list_issues(self, repo: str, state: str = "open", labels: Optional[str] = None) -> Dict:
        """List issues for a repository.

        Args:
            repo: 'owner/repo' string
            state: 'open', 'closed', or 'all'
            labels: Comma-separated label filter (optional)
        """
        owner, repo_name = self._parse_repo(repo)
        params: Dict = {"state": state, "per_page": 30}
        if labels:
            params["labels"] = labels
        data, err = self._get(f"/repos/{owner}/{repo_name}/issues", params)
        if err:
            return {"success": False, "error": err}
        issues = [i for i in (data or []) if "pull_request" not in i]
        return {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "state": state,
            "count": len(issues),
            "issues": [
                {
                    "number": i["number"],
                    "title": i["title"],
                    "author": i["user"]["login"],
                    "state": i["state"],
                    "created_at": i["created_at"][:10],
                    "labels": [l["name"] for l in i.get("labels", [])],
                    "url": i["html_url"],
                }
                for i in issues
            ],
        }

    def recent_commits(self, repo: str, limit: int = 20, branch: Optional[str] = None) -> Dict:
        """Get recent commits for a repository.

        Args:
            repo: 'owner/repo' string
            limit: Number of commits to fetch
            branch: Branch name (default: repo's default branch)
        """
        owner, repo_name = self._parse_repo(repo)
        params: Dict = {"per_page": min(limit, 100)}
        if branch:
            params["sha"] = branch
        data, err = self._get(f"/repos/{owner}/{repo_name}/commits", params)
        if err:
            return {"success": False, "error": err}
        commits = (data or [])[:limit]
        return {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "count": len(commits),
            "commits": [
                {
                    "sha": c["sha"][:7],
                    "message": c["commit"]["message"].split("\n")[0][:100],
                    "author": c["commit"]["author"]["name"],
                    "email": c["commit"]["author"]["email"],
                    "date": c["commit"]["author"]["date"][:10],
                    "url": c["html_url"],
                }
                for c in commits
            ],
        }

    def repo_stats(self, repo: str) -> Dict:
        """Get repository statistics and metadata.

        Args:
            repo: 'owner/repo' string
        """
        owner, repo_name = self._parse_repo(repo)
        data, err = self._get(f"/repos/{owner}/{repo_name}")
        if err:
            return {"success": False, "error": err}

        languages, _ = self._get(f"/repos/{owner}/{repo_name}/languages")
        contributors, _ = self._get(f"/repos/{owner}/{repo_name}/contributors", {"per_page": 10})

        return {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "description": data.get("description"),
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "open_issues": data.get("open_issues_count", 0),
            "language": data.get("language"),
            "languages": languages or {},
            "default_branch": data.get("default_branch"),
            "created_at": data.get("created_at", "")[:10],
            "updated_at": data.get("updated_at", "")[:10],
            "topics": data.get("topics", []),
            "top_contributors": [
                {"user": c["login"], "contributions": c["contributions"]}
                for c in (contributors or [])[:5]
            ],
        }

    def search_code(self, query: str, repo: Optional[str] = None) -> Dict:
        """Search code on GitHub.

        Args:
            query: Code search query
            repo: Limit search to 'owner/repo' (optional)
        """
        q = query
        if repo:
            q = f"{query} repo:{repo}"
        data, err = self._get("/search/code", {"q": q, "per_page": 10})
        if err:
            return {"success": False, "error": err}
        items = data.get("items", []) if isinstance(data, dict) else []
        return {
            "success": True,
            "query": q,
            "total_count": data.get("total_count", 0) if isinstance(data, dict) else 0,
            "results": [
                {
                    "name": i["name"],
                    "path": i["path"],
                    "repo": i["repository"]["full_name"],
                    "url": i["html_url"],
                }
                for i in items
            ],
        }

    def my_notifications(self, unread_only: bool = True) -> Dict:
        """Get your GitHub notifications."""
        if not self._token:
            return {"success": False, "error": "GITHUB_TOKEN required for notifications"}
        params = {"all": "false" if unread_only else "true", "per_page": 20}
        data, err = self._get("/notifications", params)
        if err:
            return {"success": False, "error": err}
        notifications = data or []
        return {
            "success": True,
            "unread_only": unread_only,
            "count": len(notifications),
            "notifications": [
                {
                    "id": n["id"],
                    "repo": n["repository"]["full_name"],
                    "type": n["subject"]["type"],
                    "title": n["subject"]["title"],
                    "reason": n["reason"],
                    "updated_at": n["updated_at"][:10],
                }
                for n in notifications
            ],
        }

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a GitHub action."""
        a = action.lower().strip()
        repo = kwargs.get("repo") or kwargs.get("repository")

        if "summary" in a or "week" in a or "activity" in a:
            return self.weekly_summary(repo)
        if "pr" in a or "pull" in a:
            state = kwargs.get("state", "open")
            return self.list_prs(repo or "", state)
        if "issue" in a:
            state = kwargs.get("state", "open")
            return self.list_issues(repo or "", state, kwargs.get("labels"))
        if "commit" in a:
            return self.recent_commits(repo or "", kwargs.get("limit", 20))
        if "stat" in a or "info" in a:
            return self.repo_stats(repo or "")
        if "search" in a:
            return self.search_code(kwargs.get("query") or action, repo)
        if "notification" in a:
            return self.my_notifications(kwargs.get("unread_only", True))
        if repo:
            return self.weekly_summary(repo)
        return self.weekly_summary()
