"""Plugin Marketplace Tool for browsing, installing, and managing plugins.

This tool allows users to:
- Browse and search plugins from a remote registry
- Install/uninstall plugins from GitHub
- Rate and review plugins
- Publish custom tools as plugins
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from .tool_template import BLOCKED_PATTERNS


class MarketplaceTool:
    """Tool for managing plugins from the marketplace."""

    name = "marketplace"
    description = "Browse, install, and manage plugins from the marketplace"

    # Remote registry URL
    REGISTRY_URL = "https://raw.githubusercontent.com/ElnurIbrahimov/aura-plugins/main/registry.json"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ElnurIbrahimov/aura-plugins/main/plugins"

    # Valid categories and sort options
    CATEGORIES = ["utilities", "health", "finance", "productivity", "fun", "all"]
    SORT_OPTIONS = ["downloads", "rating", "newest"]

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data"
        self.custom_tools_path = Path(__file__).parent / "custom"
        self.logs_path = self.base_path / "logs" / "marketplace"

        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.custom_tools_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # File paths
        self.installed_plugins_file = self.data_path / "installed_plugins.json"
        self.cache_file = self.data_path / "marketplace_cache.json"
        self.ratings_file = self.data_path / "plugin_ratings.json"
        self.custom_tools_registry = self.data_path / "custom_tools.json"

        # Initialize files if they don't exist
        self._init_files()

    def _init_files(self) -> None:
        """Initialize JSON files if they don't exist."""
        if not self.installed_plugins_file.exists():
            self._write_json(self.installed_plugins_file, {"plugins": []})

        if not self.cache_file.exists():
            self._write_json(self.cache_file, {"plugins": [], "last_updated": None})

        if not self.ratings_file.exists():
            self._write_json(self.ratings_file, {"ratings": {}})

    def _read_json(self, path: Path) -> dict:
        """Read JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write_json(self, path: Path, data: dict) -> None:
        """Write JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _log(self, action: str, plugin_id: str, details: str = "") -> None:
        """Log marketplace actions."""
        log_file = self.logs_path / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {action}: {plugin_id}"
        if details:
            log_entry += f" - {details}"
        log_entry += "\n"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def _fetch_registry(self) -> dict:
        """Fetch the plugin registry from GitHub."""
        try:
            response = requests.get(self.REGISTRY_URL, timeout=10)
            response.raise_for_status()
            registry = response.json()

            # Update cache
            cache_data = {
                "plugins": registry.get("plugins", []),
                "last_updated": datetime.now().isoformat()
            }
            self._write_json(self.cache_file, cache_data)

            return registry
        except requests.RequestException as e:
            # Fall back to cache
            cache = self._read_json(self.cache_file)
            if cache.get("plugins"):
                return cache
            return {"plugins": [], "error": str(e)}

    def _scan_for_dangerous_code(self, code: str) -> list[str]:
        """Scan code for dangerous patterns."""
        dangerous = []
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                dangerous.append(pattern)
        return dangerous

    def browse(self, category: str = None, sort_by: str = "downloads") -> dict:
        """Browse plugins from the marketplace.

        Args:
            category: Filter by category (utilities, health, finance, productivity, fun, all)
            sort_by: Sort by (downloads, rating, newest)

        Returns:
            Dict with success status and list of plugins
        """
        try:
            registry = self._fetch_registry()
            plugins = registry.get("plugins", [])

            if "error" in registry and not plugins:
                return {
                    "success": False,
                    "error": f"Failed to fetch registry: {registry['error']}"
                }

            # Filter by category
            if category and category.lower() != "all":
                category_lower = category.lower()
                plugins = [p for p in plugins if p.get("category", "").lower() == category_lower]

            # Sort plugins
            sort_by = sort_by.lower() if sort_by else "downloads"
            if sort_by == "downloads":
                plugins.sort(key=lambda x: x.get("downloads", 0), reverse=True)
            elif sort_by == "rating":
                plugins.sort(key=lambda x: x.get("rating", 0), reverse=True)
            elif sort_by == "newest":
                plugins.sort(key=lambda x: x.get("created", ""), reverse=True)

            # Get installed plugin IDs for comparison
            installed = self._read_json(self.installed_plugins_file)
            installed_ids = {p["id"] for p in installed.get("plugins", [])}

            # Format output
            plugin_list = []
            for p in plugins:
                plugin_list.append({
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "description": p.get("description", "")[:100],
                    "category": p.get("category"),
                    "author": p.get("author"),
                    "rating": p.get("rating", 0),
                    "downloads": p.get("downloads", 0),
                    "installed": p.get("id") in installed_ids
                })

            return {
                "success": True,
                "plugins": plugin_list,
                "total": len(plugin_list),
                "category": category or "all",
                "sort_by": sort_by
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, query: str) -> dict:
        """Search plugins by keyword.

        Args:
            query: Search keyword

        Returns:
            Dict with success status and matching plugins
        """
        try:
            if not query:
                return {"success": False, "error": "Search query is required"}

            registry = self._fetch_registry()
            plugins = registry.get("plugins", [])
            query_lower = query.lower()

            # Search in name, description, and keywords
            results = []
            for p in plugins:
                name = p.get("name", "").lower()
                description = p.get("description", "").lower()
                keywords = [k.lower() for k in p.get("keywords", [])]

                if (query_lower in name or
                    query_lower in description or
                    any(query_lower in kw for kw in keywords)):
                    results.append({
                        "id": p.get("id"),
                        "name": p.get("name"),
                        "description": p.get("description", "")[:100],
                        "category": p.get("category"),
                        "author": p.get("author"),
                        "rating": p.get("rating", 0)
                    })

            return {
                "success": True,
                "query": query,
                "results": results,
                "total": len(results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_info(self, plugin_id: str) -> dict:
        """Get full plugin details.

        Args:
            plugin_id: The plugin identifier

        Returns:
            Dict with success status and full plugin information
        """
        try:
            if not plugin_id:
                return {"success": False, "error": "Plugin ID is required"}

            registry = self._fetch_registry()
            plugins = registry.get("plugins", [])

            # Find plugin
            plugin = None
            for p in plugins:
                if p.get("id") == plugin_id:
                    plugin = p
                    break

            if not plugin:
                return {"success": False, "error": f"Plugin '{plugin_id}' not found"}

            # Check if installed
            installed = self._read_json(self.installed_plugins_file)
            installed_ids = {p["id"]: p for p in installed.get("plugins", [])}
            is_installed = plugin_id in installed_ids

            return {
                "success": True,
                "plugin": {
                    "id": plugin.get("id"),
                    "name": plugin.get("name"),
                    "description": plugin.get("description"),
                    "category": plugin.get("category"),
                    "author": plugin.get("author"),
                    "version": plugin.get("version", "1.0.0"),
                    "rating": plugin.get("rating", 0),
                    "downloads": plugin.get("downloads", 0),
                    "functions": plugin.get("functions", []),
                    "keywords": plugin.get("keywords", []),
                    "created": plugin.get("created"),
                    "repo_url": plugin.get("repo_url"),
                    "installed": is_installed,
                    "installed_version": installed_ids.get(plugin_id, {}).get("version") if is_installed else None
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def install(self, plugin_id: str) -> dict:
        """Install a plugin from the marketplace.

        Args:
            plugin_id: The plugin identifier to install

        Returns:
            Dict with success status and installation details
        """
        try:
            if not plugin_id:
                return {"success": False, "error": "Plugin ID is required"}

            # Check if already installed
            installed = self._read_json(self.installed_plugins_file)
            installed_ids = {p["id"] for p in installed.get("plugins", [])}
            if plugin_id in installed_ids:
                return {"success": False, "error": f"Plugin '{plugin_id}' is already installed"}

            # Get plugin info
            info_result = self.get_info(plugin_id)
            if not info_result.get("success"):
                return info_result

            plugin = info_result["plugin"]

            # Download plugin code from GitHub
            plugin_url = f"{self.GITHUB_RAW_BASE}/{plugin_id}/{plugin_id}.py"
            try:
                response = requests.get(plugin_url, timeout=15)
                response.raise_for_status()
                plugin_code = response.text
            except requests.RequestException as e:
                self._log("INSTALL_FAILED", plugin_id, f"Download failed: {e}")
                return {"success": False, "error": f"Failed to download plugin: {e}"}

            # Safety scan
            dangerous_patterns = self._scan_for_dangerous_code(plugin_code)
            if dangerous_patterns:
                self._log("INSTALL_BLOCKED", plugin_id, f"Dangerous code: {dangerous_patterns}")
                return {
                    "success": False,
                    "error": "Plugin contains dangerous code patterns",
                    "blocked_patterns": dangerous_patterns,
                    "requires_confirmation": True,
                    "message": "This plugin contains potentially unsafe code. Installation blocked for security."
                }

            # Show what will be installed
            functions = plugin.get("functions", [])
            confirmation_required = {
                "plugin_id": plugin_id,
                "name": plugin.get("name"),
                "author": plugin.get("author"),
                "functions": functions,
                "message": f"Installing plugin '{plugin.get('name')}' with {len(functions)} function(s): {', '.join(functions)}"
            }

            # Save plugin file
            plugin_file = self.custom_tools_path / f"{plugin_id}.py"
            with open(plugin_file, "w", encoding="utf-8") as f:
                f.write(plugin_code)

            # Determine class name from plugin code
            class_match = re.search(r'class\s+(\w+Tool)\s*[:(]', plugin_code)
            class_name = class_match.group(1) if class_match else f"{plugin_id.title().replace('_', '')}Tool"

            # Register in installed plugins
            installed_entry = {
                "id": plugin_id,
                "name": plugin.get("name"),
                "version": plugin.get("version", "1.0.0"),
                "installed_at": datetime.now().isoformat(),
                "source": "marketplace",
                "file": str(plugin_file),
                "class_name": class_name,
                "functions": functions
            }

            installed["plugins"].append(installed_entry)
            self._write_json(self.installed_plugins_file, installed)

            # Also register in custom_tools.json for agent to load
            custom_tools = self._read_json(self.custom_tools_registry)
            if "tools" not in custom_tools:
                custom_tools["tools"] = []

            custom_tool_entry = {
                "name": plugin_id,
                "class_name": class_name,
                "description": plugin.get("description", ""),
                "status": "active",
                "created": datetime.now().isoformat(),
                "file": str(plugin_file),
                "functions": functions,
                "enabled_at": datetime.now().isoformat(),
                "keywords": plugin.get("keywords", []),
                "source": "marketplace"
            }
            custom_tools["tools"].append(custom_tool_entry)
            self._write_json(self.custom_tools_registry, custom_tools)

            self._log("INSTALLED", plugin_id, f"Version {plugin.get('version', '1.0.0')}")

            return {
                "success": True,
                "message": f"Successfully installed '{plugin.get('name')}'",
                "plugin_id": plugin_id,
                "version": plugin.get("version", "1.0.0"),
                "functions": functions,
                "note": "Restart the agent to load the new plugin"
            }
        except Exception as e:
            self._log("INSTALL_ERROR", plugin_id, str(e))
            return {"success": False, "error": str(e)}

    def uninstall(self, plugin_id: str) -> dict:
        """Uninstall a plugin.

        Args:
            plugin_id: The plugin identifier to uninstall

        Returns:
            Dict with success status
        """
        try:
            if not plugin_id:
                return {"success": False, "error": "Plugin ID is required"}

            # Check if installed
            installed = self._read_json(self.installed_plugins_file)
            plugin_entry = None
            for p in installed.get("plugins", []):
                if p["id"] == plugin_id:
                    plugin_entry = p
                    break

            if not plugin_entry:
                return {"success": False, "error": f"Plugin '{plugin_id}' is not installed"}

            # Remove plugin file
            plugin_file = Path(plugin_entry.get("file", ""))
            if plugin_file.exists():
                plugin_file.unlink()

            # Remove from installed plugins
            installed["plugins"] = [p for p in installed["plugins"] if p["id"] != plugin_id]
            self._write_json(self.installed_plugins_file, installed)

            # Remove from custom_tools.json
            custom_tools = self._read_json(self.custom_tools_registry)
            custom_tools["tools"] = [t for t in custom_tools.get("tools", []) if t.get("name") != plugin_id]
            self._write_json(self.custom_tools_registry, custom_tools)

            self._log("UNINSTALLED", plugin_id)

            return {
                "success": True,
                "message": f"Successfully uninstalled '{plugin_id}'",
                "plugin_id": plugin_id,
                "note": "Restart the agent to fully remove the plugin"
            }
        except Exception as e:
            self._log("UNINSTALL_ERROR", plugin_id, str(e))
            return {"success": False, "error": str(e)}

    def publish(self, tool_name: str) -> dict:
        """Package a custom tool for publishing to the marketplace.

        Args:
            tool_name: Name of the custom tool to package

        Returns:
            Dict with success status and publishing instructions
        """
        try:
            if not tool_name:
                return {"success": False, "error": "Tool name is required"}

            # Find the tool in custom_tools.json
            custom_tools = self._read_json(self.custom_tools_registry)
            tool_entry = None
            for t in custom_tools.get("tools", []):
                if t.get("name") == tool_name:
                    tool_entry = t
                    break

            if not tool_entry:
                return {"success": False, "error": f"Custom tool '{tool_name}' not found"}

            # Read the tool source code
            tool_file = Path(tool_entry.get("file", ""))
            if not tool_file.exists():
                return {"success": False, "error": f"Tool file not found: {tool_file}"}

            with open(tool_file, "r", encoding="utf-8") as f:
                tool_code = f.read()

            # Create plugin folder structure
            publish_dir = self.base_path / "plugins" / tool_name
            publish_dir.mkdir(parents=True, exist_ok=True)

            # Copy tool file
            shutil.copy(tool_file, publish_dir / f"{tool_name}.py")

            # Copy test file if exists
            test_file = self.custom_tools_path / "tests" / f"test_{tool_name}.py"
            if test_file.exists():
                (publish_dir / "tests").mkdir(exist_ok=True)
                shutil.copy(test_file, publish_dir / "tests" / f"test_{tool_name}.py")

            # Generate plugin.json
            plugin_json = {
                "id": tool_name,
                "name": tool_entry.get("description", tool_name).split('.')[0],
                "description": tool_entry.get("description", ""),
                "category": "utilities",  # Default category
                "author": "local",
                "version": "1.0.0",
                "functions": tool_entry.get("functions", []),
                "keywords": tool_entry.get("keywords", []),
                "created": datetime.now().isoformat()
            }
            self._write_json(publish_dir / "plugin.json", plugin_json)

            # Generate README.md
            readme_content = f"""# {plugin_json['name']}

{plugin_json['description']}

## Functions

{chr(10).join('- ' + f for f in plugin_json['functions'])}

## Installation

```
marketplace install {tool_name}
```

## Keywords

{', '.join(plugin_json['keywords'])}
"""
            with open(publish_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)

            self._log("PACKAGED", tool_name, str(publish_dir))

            return {
                "success": True,
                "message": f"Plugin packaged successfully",
                "plugin_id": tool_name,
                "output_dir": str(publish_dir),
                "files": [
                    f"{tool_name}.py",
                    "plugin.json",
                    "README.md",
                    "tests/" if test_file.exists() else None
                ],
                "next_steps": [
                    f"1. Review files in: {publish_dir}",
                    "2. Update plugin.json with correct category and author",
                    "3. Push to github.com/ElnurIbrahimov/aura-plugins",
                    "4. Add entry to registry.json"
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rate(self, plugin_id: str, stars: int) -> dict:
        """Rate a plugin (1-5 stars).

        Args:
            plugin_id: The plugin identifier
            stars: Rating from 1 to 5

        Returns:
            Dict with success status
        """
        try:
            if not plugin_id:
                return {"success": False, "error": "Plugin ID is required"}

            if not isinstance(stars, int) or stars < 1 or stars > 5:
                return {"success": False, "error": "Rating must be between 1 and 5 stars"}

            # Load ratings
            ratings = self._read_json(self.ratings_file)
            if "ratings" not in ratings:
                ratings["ratings"] = {}

            # Save rating
            ratings["ratings"][plugin_id] = {
                "stars": stars,
                "rated_at": datetime.now().isoformat()
            }
            self._write_json(self.ratings_file, ratings)

            self._log("RATED", plugin_id, f"{stars} stars")

            return {
                "success": True,
                "message": f"Rated '{plugin_id}' with {stars} star(s)",
                "plugin_id": plugin_id,
                "stars": stars
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def my_plugins(self) -> dict:
        """List all installed marketplace plugins.

        Returns:
            Dict with success status and list of installed plugins
        """
        try:
            installed = self._read_json(self.installed_plugins_file)
            plugins = installed.get("plugins", [])

            plugin_list = []
            for p in plugins:
                plugin_list.append({
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "version": p.get("version"),
                    "installed_at": p.get("installed_at"),
                    "source": p.get("source"),
                    "functions": p.get("functions", [])
                })

            return {
                "success": True,
                "plugins": plugin_list,
                "total": len(plugin_list)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update(self, plugin_id: str) -> dict:
        """Check for and install plugin updates.

        Args:
            plugin_id: The plugin identifier to update

        Returns:
            Dict with success status and update information
        """
        try:
            if not plugin_id:
                return {"success": False, "error": "Plugin ID is required"}

            # Check if installed
            installed = self._read_json(self.installed_plugins_file)
            plugin_entry = None
            for p in installed.get("plugins", []):
                if p["id"] == plugin_id:
                    plugin_entry = p
                    break

            if not plugin_entry:
                return {"success": False, "error": f"Plugin '{plugin_id}' is not installed"}

            current_version = plugin_entry.get("version", "1.0.0")

            # Get latest version from registry
            info_result = self.get_info(plugin_id)
            if not info_result.get("success"):
                return info_result

            latest_version = info_result["plugin"].get("version", "1.0.0")

            # Compare versions (simple string comparison)
            if latest_version <= current_version:
                return {
                    "success": True,
                    "message": f"Plugin '{plugin_id}' is up to date",
                    "plugin_id": plugin_id,
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "update_available": False
                }

            # Uninstall old version and install new
            uninstall_result = self.uninstall(plugin_id)
            if not uninstall_result.get("success"):
                return uninstall_result

            install_result = self.install(plugin_id)
            if not install_result.get("success"):
                return install_result

            self._log("UPDATED", plugin_id, f"{current_version} -> {latest_version}")

            return {
                "success": True,
                "message": f"Updated '{plugin_id}' from {current_version} to {latest_version}",
                "plugin_id": plugin_id,
                "previous_version": current_version,
                "new_version": latest_version,
                "update_available": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str) -> dict:
        """Execute a marketplace action.

        Args:
            action: The action string to parse and execute

        Returns:
            Result dictionary with success status and data
        """
        action_lower = action.lower()

        # Browse plugins
        if "browse" in action_lower or "list plugin" in action_lower:
            category = self._extract_category(action)
            sort_by = self._extract_sort(action)
            return self.browse(category=category, sort_by=sort_by)

        # Search plugins
        if "search" in action_lower:
            query = self._extract_query(action)
            if query:
                return self.search(query)
            return {"success": False, "error": "Please provide a search query"}

        # Get plugin info
        if "info" in action_lower or "details" in action_lower:
            plugin_id = self._extract_plugin_id(action)
            if plugin_id:
                return self.get_info(plugin_id)
            return {"success": False, "error": "Please provide a plugin ID"}

        # Install plugin
        if "install" in action_lower:
            plugin_id = self._extract_plugin_id(action)
            if plugin_id:
                return self.install(plugin_id)
            return {"success": False, "error": "Please provide a plugin ID to install"}

        # Uninstall plugin
        if "uninstall" in action_lower or "remove" in action_lower:
            plugin_id = self._extract_plugin_id(action)
            if plugin_id:
                return self.uninstall(plugin_id)
            return {"success": False, "error": "Please provide a plugin ID to uninstall"}

        # Publish tool
        if "publish" in action_lower or "share" in action_lower:
            tool_name = self._extract_tool_name(action)
            if tool_name:
                return self.publish(tool_name)
            return {"success": False, "error": "Please provide a tool name to publish"}

        # Rate plugin
        if "rate" in action_lower:
            plugin_id = self._extract_plugin_id(action)
            stars = self._extract_stars(action)
            if plugin_id and stars:
                return self.rate(plugin_id, stars)
            return {"success": False, "error": "Please provide plugin ID and rating (1-5 stars)"}

        # My plugins
        if "my plugin" in action_lower or "installed" in action_lower:
            return self.my_plugins()

        # Update plugin
        if "update" in action_lower:
            plugin_id = self._extract_plugin_id(action)
            if plugin_id:
                return self.update(plugin_id)
            return {"success": False, "error": "Please provide a plugin ID to update"}

        return {"success": False, "error": f"Unknown marketplace action: {action}"}

    def _extract_category(self, action: str) -> Optional[str]:
        """Extract category from action string."""
        action_lower = action.lower()
        for cat in self.CATEGORIES:
            if cat in action_lower:
                return cat
        return None

    def _extract_sort(self, action: str) -> str:
        """Extract sort option from action string."""
        action_lower = action.lower()
        for sort in self.SORT_OPTIONS:
            if sort in action_lower:
                return sort
        return "downloads"

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Remove command words
        query = re.sub(
            r'\b(search|find|look|for|plugin|plugins|marketplace)\b',
            '', action, flags=re.IGNORECASE
        )
        return query.strip() or None

    def _extract_plugin_id(self, action: str) -> Optional[str]:
        """Extract plugin ID from action string."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            return quoted[0]

        # Look for snake_case identifiers
        ids = re.findall(r'\b([a-z][a-z0-9_]+)\b', action.lower())
        # Filter out common words
        exclude = {'install', 'uninstall', 'update', 'info', 'rate', 'plugin', 'marketplace',
                   'browse', 'search', 'publish', 'share', 'remove', 'details', 'stars'}
        for id_ in ids:
            if id_ not in exclude and len(id_) > 3:
                return id_

        return None

    def _extract_tool_name(self, action: str) -> Optional[str]:
        """Extract tool name from action string."""
        return self._extract_plugin_id(action)

    def _extract_stars(self, action: str) -> Optional[int]:
        """Extract star rating from action string."""
        # Look for number followed by "star"
        match = re.search(r'(\d)\s*star', action.lower())
        if match:
            return int(match.group(1))

        # Look for standalone number 1-5
        numbers = re.findall(r'\b([1-5])\b', action)
        if numbers:
            return int(numbers[-1])

        return None
