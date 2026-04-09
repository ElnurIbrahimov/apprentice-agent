"""Deploy Tool — deploy web projects to Vercel, Netlify, GitHub Pages, or local preview.

Auto-detects project type (Next.js, Vite, React, static HTML) and available
platform CLIs, then deploys with a single action call.

No external Python dependencies — uses subprocess to invoke platform CLIs
and the stdlib http.server for local preview.
"""

import http.server
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPLOY_TIMEOUT = 120  # 2 minutes max for deploy commands
BUILD_TIMEOUT = 180   # 3 minutes max for build commands
CLI_CHECK_TIMEOUT = 10

# Framework detection patterns (filename -> framework name)
_FRAMEWORK_MARKERS = {
    "next.config.js": "nextjs",
    "next.config.ts": "nextjs",
    "next.config.mjs": "nextjs",
    "vite.config.js": "vite",
    "vite.config.ts": "vite",
    "vite.config.mjs": "vite",
    "gatsby-config.js": "gatsby",
    "gatsby-config.ts": "gatsby",
    "nuxt.config.js": "nuxt",
    "nuxt.config.ts": "nuxt",
    "astro.config.mjs": "astro",
    "astro.config.ts": "astro",
    "svelte.config.js": "sveltekit",
    "remix.config.js": "remix",
    "angular.json": "angular",
}

# Platform recommendation by framework
_PLATFORM_PREFERENCE = {
    "nextjs": ["vercel", "netlify"],
    "vite": ["vercel", "netlify", "gh-pages"],
    "gatsby": ["netlify", "vercel"],
    "nuxt": ["vercel", "netlify"],
    "astro": ["netlify", "vercel"],
    "sveltekit": ["vercel", "netlify"],
    "remix": ["vercel", "netlify"],
    "angular": ["vercel", "netlify"],
    "react": ["vercel", "netlify", "gh-pages"],
    "static": ["netlify", "gh-pages", "vercel", "local"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: List[str], cwd: str = ".", timeout: int = DEPLOY_TIMEOUT,
         env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run a subprocess and return structured result."""
    resolved = str(Path(cwd).resolve())
    if not Path(resolved).exists():
        return {"success": False, "error": f"Path does not exist: {cwd}"}

    # Use allowlist (not denylist) — only pass safe env vars to child processes
    from aura.tools.shell_executor import _get_sanitized_env
    run_env = _get_sanitized_env()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            args,
            cwd=resolved,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return {"success": True, "output": stdout, "stderr": stderr}
        return {
            "success": False,
            "error": stderr or stdout or f"Command exited with code {result.returncode}",
            "output": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "error": f"Command not found: {args[0]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _which(cmd: str) -> Optional[str]:
    """Check if a command exists on PATH."""
    return shutil.which(cmd)


# ---------------------------------------------------------------------------
# DeployTool
# ---------------------------------------------------------------------------

class DeployTool:
    """Deploy web projects to Vercel, Netlify, GitHub Pages, or local preview."""

    name = "deploy"
    description = "Deploy web projects to Vercel, Netlify, GitHub Pages, or local preview"

    def __init__(self):
        self._local_server: Optional[threading.Thread] = None
        self._local_httpd: Optional[http.server.HTTPServer] = None
        self._local_port: Optional[int] = None
        self._cli_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    #  CLI Detection
    # ------------------------------------------------------------------ #

    def _check_cli(self, name: str) -> bool:
        """Check if a CLI tool is installed. Results are cached."""
        if name in self._cli_cache:
            return self._cli_cache[name]

        found = _which(name) is not None
        if not found:
            # Try npx as fallback for node-based CLIs
            if name in ("vercel", "netlify"):
                r = _run(["npx", name, "--version"], timeout=CLI_CHECK_TIMEOUT)
                found = r["success"]

        self._cli_cache[name] = found
        return found

    def available_platforms(self) -> Dict[str, Any]:
        """Detect which deployment platforms are available."""
        platforms = {}

        # Vercel
        has_vercel = self._check_cli("vercel")
        platforms["vercel"] = {
            "available": has_vercel,
            "install": "npm i -g vercel" if not has_vercel else None,
        }

        # Netlify
        has_netlify = self._check_cli("netlify")
        platforms["netlify"] = {
            "available": has_netlify,
            "install": "npm i -g netlify-cli" if not has_netlify else None,
        }

        # Git (needed for GitHub Pages)
        has_git = self._check_cli("git")
        platforms["gh-pages"] = {
            "available": has_git,
            "install": "Install git from https://git-scm.com" if not has_git else None,
        }

        # Local preview is always available
        platforms["local"] = {"available": True, "install": None}

        return platforms

    # ------------------------------------------------------------------ #
    #  Project Detection
    # ------------------------------------------------------------------ #

    def detect_project(self, path: str) -> Dict[str, Any]:
        """Detect the project type and framework in a directory."""
        p = Path(path).resolve()
        if not p.exists():
            return {"success": False, "error": f"Path does not exist: {path}"}
        if not p.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}

        info: Dict[str, Any] = {
            "success": True,
            "path": str(p),
            "framework": None,
            "has_package_json": False,
            "has_build_script": False,
            "has_index_html": False,
            "build_command": None,
            "output_dir": None,
        }

        # Check framework config files
        for marker, framework in _FRAMEWORK_MARKERS.items():
            if (p / marker).exists():
                info["framework"] = framework
                break

        # Check package.json
        pkg_path = p / "package.json"
        if pkg_path.exists():
            info["has_package_json"] = True
            try:
                with open(pkg_path, "r", encoding="utf-8") as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                if "build" in scripts:
                    info["has_build_script"] = True
                    info["build_command"] = "npm run build"

                # Detect framework from dependencies if not already detected
                if not info["framework"]:
                    deps = {
                        **pkg.get("dependencies", {}),
                        **pkg.get("devDependencies", {}),
                    }
                    if "next" in deps:
                        info["framework"] = "nextjs"
                    elif "vite" in deps:
                        info["framework"] = "vite"
                    elif "gatsby" in deps:
                        info["framework"] = "gatsby"
                    elif "nuxt" in deps:
                        info["framework"] = "nuxt"
                    elif "astro" in deps:
                        info["framework"] = "astro"
                    elif "@sveltejs/kit" in deps:
                        info["framework"] = "sveltekit"
                    elif "@remix-run/react" in deps:
                        info["framework"] = "remix"
                    elif "react" in deps:
                        info["framework"] = "react"
                    elif "vue" in deps:
                        info["framework"] = "vue"

            except (json.JSONDecodeError, IOError):
                pass

        # Check for index.html (static site)
        if (p / "index.html").exists():
            info["has_index_html"] = True
            if not info["framework"]:
                info["framework"] = "static"

        # Determine output directory
        fw = info["framework"]
        if fw == "nextjs":
            info["output_dir"] = ".next"
        elif fw in ("vite", "react", "vue"):
            info["output_dir"] = "dist"
        elif fw == "gatsby":
            info["output_dir"] = "public"
        elif fw == "astro":
            info["output_dir"] = "dist"
        elif fw == "angular":
            info["output_dir"] = "dist"
        elif fw == "static":
            info["output_dir"] = "."

        return info

    def suggest_platform(self, path: str) -> Dict[str, Any]:
        """Suggest the best deployment platform for a project."""
        project = self.detect_project(path)
        if not project.get("success"):
            return project

        platforms = self.available_platforms()
        framework = project.get("framework") or "static"
        preferred = _PLATFORM_PREFERENCE.get(framework, ["vercel", "netlify", "local"])

        suggestions = []
        for plat in preferred:
            info = platforms.get(plat, {})
            suggestions.append({
                "platform": plat,
                "available": info.get("available", False),
                "install": info.get("install"),
            })

        # Best = first available in preference order
        best = next((s for s in suggestions if s["available"]), suggestions[-1] if suggestions else None)

        return {
            "success": True,
            "framework": framework,
            "recommended": best["platform"] if best else "local",
            "suggestions": suggestions,
        }

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def _run_build(self, path: str) -> Dict[str, Any]:
        """Run the build step if a build script exists."""
        project = self.detect_project(path)
        if not project.get("has_build_script"):
            return {"success": True, "skipped": True, "message": "No build script found, skipping build"}

        # Install deps first if node_modules missing
        p = Path(path).resolve()
        if project["has_package_json"] and not (p / "node_modules").exists():
            logger.info("[Deploy] Installing dependencies...")
            install_cmd = ["npm", "install"]
            if _which("pnpm"):
                install_cmd = ["pnpm", "install"]
            elif _which("yarn"):
                install_cmd = ["yarn", "install"]
            install = _run(install_cmd, cwd=path, timeout=BUILD_TIMEOUT)
            if not install["success"]:
                return {
                    "success": False,
                    "error": f"Dependency install failed: {install.get('error', '')}",
                    "suggestion": "Try running 'npm install' manually first.",
                }

        logger.info("[Deploy] Running build...")
        build = _run(["npm", "run", "build"], cwd=path, timeout=BUILD_TIMEOUT)
        if not build["success"]:
            return {
                "success": False,
                "error": f"Build failed: {build.get('error', '')}",
                "output": build.get("output", ""),
                "suggestion": "Check your build script in package.json. Try 'npm run build' manually.",
            }

        return {"success": True, "skipped": False, "message": "Build completed successfully"}

    # ------------------------------------------------------------------ #
    #  Vercel
    # ------------------------------------------------------------------ #

    def deploy_vercel(self, path: str, production: bool = False, token: Optional[str] = None) -> Dict[str, Any]:
        """Deploy to Vercel via CLI."""
        if not self._check_cli("vercel"):
            return {
                "success": False,
                "error": "Vercel CLI not found.",
                "suggestion": "Install it with: npm i -g vercel\nThen run: vercel login",
            }

        cmd = ["vercel"]
        if production:
            cmd.append("--prod")
        cmd.append("--yes")  # Skip prompts

        if token:
            cmd.extend(["--token", token])

        env_token = os.environ.get("VERCEL_TOKEN")
        env = {}
        if env_token and not token:
            env["VERCEL_TOKEN"] = env_token

        logger.info("[Deploy] Deploying to Vercel%s...", " (production)" if production else " (preview)")
        result = _run(cmd, cwd=path, timeout=DEPLOY_TIMEOUT, env=env or None)

        if not result["success"]:
            error = result.get("error", "")
            suggestion = None
            if "not linked" in error.lower() or "no project" in error.lower():
                suggestion = "Run 'vercel' in your project directory first to link it."
            elif "unauthorized" in error.lower() or "login" in error.lower():
                suggestion = "Run 'vercel login' to authenticate."
            return {
                "success": False,
                "error": error,
                "suggestion": suggestion,
            }

        # Parse the deployment URL from output
        output = result.get("output", "")
        url = self._parse_vercel_url(output)

        return {
            "success": True,
            "platform": "vercel",
            "production": production,
            "url": url,
            "output": output,
        }

    def _parse_vercel_url(self, output: str) -> Optional[str]:
        """Extract the deployment URL from Vercel CLI output."""
        # Vercel outputs the URL on its own line, typically https://*.vercel.app
        for line in output.split("\n"):
            line = line.strip()
            if re.match(r'^https://[\w.-]+\.vercel\.app', line):
                return line
            # Also match production URLs
            if re.match(r'^https://[\w.-]+', line) and "vercel" in line.lower():
                return line
        # Fallback: any https:// URL
        urls = re.findall(r'https://[\w.-]+(?:/[\w.-]*)*', output)
        return urls[0] if urls else None

    def list_vercel_deployments(self, path: str = ".") -> Dict[str, Any]:
        """List Vercel deployments for the current project."""
        if not self._check_cli("vercel"):
            return {"success": False, "error": "Vercel CLI not found."}

        result = _run(["vercel", "ls"], cwd=path, timeout=30)
        if not result["success"]:
            return result
        return {"success": True, "output": result.get("output", "")}

    # ------------------------------------------------------------------ #
    #  Netlify
    # ------------------------------------------------------------------ #

    def deploy_netlify(self, path: str, production: bool = False,
                       site_id: Optional[str] = None, dir_: Optional[str] = None) -> Dict[str, Any]:
        """Deploy to Netlify via CLI."""
        if not self._check_cli("netlify"):
            return {
                "success": False,
                "error": "Netlify CLI not found.",
                "suggestion": "Install it with: npm i -g netlify-cli\nThen run: netlify login",
            }

        # Build first
        build_result = self._run_build(path)
        if not build_result["success"]:
            return build_result

        # Determine deploy directory
        deploy_dir = dir_
        if not deploy_dir:
            project = self.detect_project(path)
            deploy_dir = project.get("output_dir", ".")

        cmd = ["netlify", "deploy"]
        if production:
            cmd.append("--prod")
        if site_id:
            cmd.extend(["--site", site_id])
        if deploy_dir and deploy_dir != ".":
            cmd.extend(["--dir", deploy_dir])

        logger.info("[Deploy] Deploying to Netlify%s...", " (production)" if production else " (draft)")
        result = _run(cmd, cwd=path, timeout=DEPLOY_TIMEOUT)

        if not result["success"]:
            error = result.get("error", "")
            suggestion = None
            if "not linked" in error.lower():
                suggestion = "Run 'netlify init' or 'netlify link' in your project directory."
            elif "unauthorized" in error.lower() or "login" in error.lower():
                suggestion = "Run 'netlify login' to authenticate."
            return {"success": False, "error": error, "suggestion": suggestion}

        output = result.get("output", "")
        url = self._parse_netlify_url(output)

        return {
            "success": True,
            "platform": "netlify",
            "production": production,
            "url": url,
            "output": output,
        }

    def _parse_netlify_url(self, output: str) -> Optional[str]:
        """Extract the deployment URL from Netlify CLI output."""
        # Netlify outputs: Website draft URL: https://...
        # Or: Website URL: https://...
        for line in output.split("\n"):
            if "url:" in line.lower():
                match = re.search(r'https://[\w.-]+(?:\.netlify\.app|\.netlify\.com)[\w/.-]*', line)
                if match:
                    return match.group(0)
        # Fallback
        urls = re.findall(r'https://[\w.-]+\.netlify\.(?:app|com)[\w/.-]*', output)
        return urls[0] if urls else None

    def list_netlify_sites(self) -> Dict[str, Any]:
        """List Netlify sites."""
        if not self._check_cli("netlify"):
            return {"success": False, "error": "Netlify CLI not found."}

        result = _run(["netlify", "sites:list"], timeout=30)
        if not result["success"]:
            return result
        return {"success": True, "output": result.get("output", "")}

    # ------------------------------------------------------------------ #
    #  GitHub Pages
    # ------------------------------------------------------------------ #

    def deploy_gh_pages(self, path: str, repo: Optional[str] = None,
                        branch: str = "gh-pages", build_first: bool = True) -> Dict[str, Any]:
        """Deploy to GitHub Pages by pushing to the gh-pages branch."""
        if not self._check_cli("git"):
            return {
                "success": False,
                "error": "Git not found.",
                "suggestion": "Install git from https://git-scm.com",
            }

        p = Path(path).resolve()

        # Build if needed
        if build_first:
            build_result = self._run_build(path)
            if not build_result["success"]:
                return build_result

        # Determine the output directory to deploy
        project = self.detect_project(path)
        output_dir = project.get("output_dir", ".")
        deploy_path = p / output_dir if output_dir != "." else p

        if not deploy_path.exists():
            return {
                "success": False,
                "error": f"Build output directory not found: {deploy_path}",
                "suggestion": "Run the build step first, or check your output directory.",
            }

        # Check if we're in a git repo
        git_check = _run(["git", "rev-parse", "--git-dir"], cwd=path, timeout=5)
        if not git_check["success"]:
            return {
                "success": False,
                "error": "Not a git repository.",
                "suggestion": "Initialize with 'git init' and add a remote: git remote add origin <url>",
            }

        # Get the remote URL
        if not repo:
            remote_result = _run(["git", "remote", "get-url", "origin"], cwd=path, timeout=5)
            if remote_result["success"]:
                repo = remote_result.get("output", "").strip()
            else:
                return {
                    "success": False,
                    "error": "No remote 'origin' configured and no repo URL provided.",
                    "suggestion": "Add a remote: git remote add origin https://github.com/user/repo.git",
                }

        # Use git subtree or manual approach
        # We'll use the ghp-import approach: create an orphan branch with just the build output
        logger.info("[Deploy] Deploying to GitHub Pages (branch: %s)...", branch)

        # If deploying from a subdirectory, use subtree push
        if output_dir != ".":
            # Add and commit build output first
            _run(["git", "add", "-f", output_dir], cwd=path, timeout=10)
            _run(["git", "commit", "-m", "build: deploy to gh-pages", "--allow-empty"],
                 cwd=path, timeout=10)

            # Push subtree
            result = _run(
                ["git", "subtree", "push", "--prefix", output_dir, "origin", branch],
                cwd=path, timeout=DEPLOY_TIMEOUT,
            )
        else:
            # Push the whole repo to gh-pages
            result = _run(
                ["git", "push", "origin", f"HEAD:{branch}"],
                cwd=path, timeout=DEPLOY_TIMEOUT,
            )

        if not result["success"]:
            error = result.get("error", "")
            suggestion = None
            if "authentication" in error.lower() or "permission" in error.lower():
                suggestion = "Check your git credentials or SSH keys."
            elif "rejected" in error.lower():
                suggestion = "The remote branch may have diverged. Try: git push -f origin HEAD:gh-pages"
            return {"success": False, "error": error, "suggestion": suggestion}

        # Build the GitHub Pages URL
        gh_url = self._build_gh_pages_url(repo)

        return {
            "success": True,
            "platform": "gh-pages",
            "branch": branch,
            "url": gh_url,
            "output": result.get("output", "") or result.get("stderr", ""),
            "note": "GitHub Pages may take 1-2 minutes to update.",
        }

    def _build_gh_pages_url(self, repo: Optional[str]) -> Optional[str]:
        """Construct the GitHub Pages URL from a repo URL."""
        if not repo:
            return None
        # https://github.com/user/repo.git -> https://user.github.io/repo
        match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', repo)
        if match:
            user, name = match.group(1), match.group(2)
            return f"https://{user}.github.io/{name}"
        return None

    # ------------------------------------------------------------------ #
    #  Local Preview Server
    # ------------------------------------------------------------------ #

    def serve(self, path: str, port: int = 3000) -> Dict[str, Any]:
        """Start a local HTTP server for quick preview."""
        if self._local_httpd is not None:
            return {
                "success": False,
                "error": f"Server already running on port {self._local_port}.",
                "suggestion": "Call stop_server() first, or use a different port.",
            }

        p = Path(path).resolve()
        if not p.exists():
            return {"success": False, "error": f"Path does not exist: {path}"}

        # Determine what to serve
        project = self.detect_project(str(p))
        serve_dir = p
        output = project.get("output_dir")
        if output and output != "." and (p / output).exists():
            serve_dir = p / output

        # If it's a framework project, suggest using the dev server instead
        framework = project.get("framework")
        if framework and framework not in ("static",):
            has_build = (serve_dir != p) and serve_dir.exists()
            if not has_build:
                return {
                    "success": False,
                    "error": f"This is a {framework} project. The build output isn't available.",
                    "suggestion": "Run 'npm run build' first, or use 'npm run dev' for development.",
                }

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(serve_dir), **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress request logs

        try:
            httpd = http.server.HTTPServer(("127.0.0.1", port), QuietHandler)
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                return {
                    "success": False,
                    "error": f"Port {port} is already in use.",
                    "suggestion": f"Try a different port, or stop whatever is using port {port}.",
                }
            return {"success": False, "error": str(e)}

        self._local_httpd = httpd
        self._local_port = port

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self._local_server = thread

        url = f"http://127.0.0.1:{port}"
        logger.info("[Deploy] Local preview server started at %s (serving %s)", url, serve_dir)

        return {
            "success": True,
            "platform": "local",
            "url": url,
            "serving": str(serve_dir),
            "port": port,
            "message": f"Local preview running at {url}",
        }

    def stop_server(self) -> Dict[str, Any]:
        """Stop the local preview server."""
        if self._local_httpd is None:
            return {"success": False, "error": "No local server is running."}

        try:
            self._local_httpd.shutdown()
            self._local_httpd.server_close()
        except Exception as e:
            logger.warning("[Deploy] Error stopping server: %s", e)

        port = self._local_port
        self._local_httpd = None
        self._local_server = None
        self._local_port = None

        return {
            "success": True,
            "message": f"Local server on port {port} stopped.",
        }

    # ------------------------------------------------------------------ #
    #  Unified Deploy
    # ------------------------------------------------------------------ #

    def deploy(self, path: str, platform: Optional[str] = None,
               production: bool = False, **kwargs) -> Dict[str, Any]:
        """Deploy a project to the specified (or auto-detected) platform.

        Args:
            path: Project directory path.
            platform: 'vercel', 'netlify', 'gh-pages', or 'local'. Auto-detected if None.
            production: If True, deploy to production (not preview/draft).
            **kwargs: Platform-specific options (token, site_id, repo, port, etc.)
        """
        # Validate path
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            return {"success": False, "error": f"Invalid project path: {path}"}

        # Auto-detect platform
        if not platform:
            suggestion = self.suggest_platform(str(p))
            platform = suggestion.get("recommended", "local")
            logger.info("[Deploy] Auto-selected platform: %s", platform)

        platform = platform.lower().strip()

        if platform == "vercel":
            return self.deploy_vercel(str(p), production=production,
                                      token=kwargs.get("token"))
        elif platform == "netlify":
            return self.deploy_netlify(str(p), production=production,
                                       site_id=kwargs.get("site_id"),
                                       dir_=kwargs.get("dir"))
        elif platform in ("gh-pages", "github-pages", "github"):
            return self.deploy_gh_pages(str(p), repo=kwargs.get("repo"),
                                        branch=kwargs.get("branch", "gh-pages"))
        elif platform in ("local", "preview"):
            return self.serve(str(p), port=kwargs.get("port", 3000))
        else:
            return {
                "success": False,
                "error": f"Unknown platform: {platform}",
                "suggestion": "Supported platforms: vercel, netlify, gh-pages, local",
            }

    # ------------------------------------------------------------------ #
    #  Status
    # ------------------------------------------------------------------ #

    def status(self, path: str = ".") -> Dict[str, Any]:
        """Get deployment status overview for a project."""
        project = self.detect_project(path)
        platforms = self.available_platforms()
        suggestion = self.suggest_platform(path) if project.get("success") else {}

        server_info = None
        if self._local_httpd is not None:
            server_info = {
                "running": True,
                "url": f"http://127.0.0.1:{self._local_port}",
                "port": self._local_port,
            }

        return {
            "success": True,
            "project": project,
            "platforms": platforms,
            "recommended": suggestion.get("recommended"),
            "local_server": server_info,
        }

    # ------------------------------------------------------------------ #
    #  execute() — standard Aura tool interface
    # ------------------------------------------------------------------ #

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a deploy action.

        Args:
            action: One of 'deploy', 'preview', 'list', 'status', 'detect',
                    'suggest', 'stop', 'build', 'platforms'.
            **kwargs: Action-specific arguments:
                - path: Project directory (default '.')
                - platform: 'vercel', 'netlify', 'gh-pages', 'local'
                - production: bool (default False)
                - port: int for local server (default 3000)
                - token: Vercel token override
                - site_id: Netlify site ID
                - repo: GitHub repo URL for gh-pages
                - branch: gh-pages branch name
                - dir: Netlify deploy directory override
        """
        a = action.lower().strip()
        path = kwargs.get("path", ".")

        # Deploy
        if a in ("deploy", "push", "ship"):
            platform = kwargs.get("platform")
            production = kwargs.get("production", False)
            return self.deploy(path, platform=platform, production=production, **kwargs)

        # Preview (local server)
        if a in ("preview", "serve", "local"):
            port = kwargs.get("port", 3000)
            return self.serve(path, port=port)

        # Stop local server
        if a in ("stop", "stop_server", "kill"):
            return self.stop_server()

        # Build only
        if a in ("build",):
            return self._run_build(path)

        # List deployments
        if a in ("list", "ls", "deployments"):
            platform = kwargs.get("platform", "").lower()
            if platform == "netlify":
                return self.list_netlify_sites()
            if platform == "vercel":
                return self.list_vercel_deployments(path)
            # Try both
            results = {}
            if self._check_cli("vercel"):
                results["vercel"] = self.list_vercel_deployments(path)
            if self._check_cli("netlify"):
                results["netlify"] = self.list_netlify_sites()
            if not results:
                return {"success": False, "error": "No deployment platform CLIs found."}
            return {"success": True, **results}

        # Status
        if a in ("status", "info"):
            return self.status(path)

        # Detect project
        if a in ("detect", "detect_project", "scan"):
            return self.detect_project(path)

        # Suggest platform
        if a in ("suggest", "recommend"):
            return self.suggest_platform(path)

        # Available platforms
        if a in ("platforms", "available"):
            return self.available_platforms()

        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "hint": "Supported actions: deploy, preview, stop, build, list, status, detect, suggest, platforms",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_deploy_tool: Optional[DeployTool] = None


def get_deploy_tool() -> DeployTool:
    """Get or create the singleton DeployTool instance."""
    global _deploy_tool
    if _deploy_tool is None:
        _deploy_tool = DeployTool()
    return _deploy_tool
