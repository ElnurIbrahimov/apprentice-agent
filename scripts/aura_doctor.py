#!/usr/bin/env python
"""Aura Setup Wizard & Health Check (aura-doctor).

Usage:
    python scripts/aura_doctor.py           # Check system health
    python scripts/aura_doctor.py --setup   # Interactive setup wizard
    python scripts/aura_doctor.py --quick   # Brief single-line status

Checks environment, dependencies, services, and model availability.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("aura_doctor")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def check_env_file() -> tuple[bool, str]:
    """Verify .env exists and has required keys."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return False, ".env not found. Copy .env.example to .env: cp .env.example .env"

    required = ["OLLAMA_API_KEY", "AURA_API_KEY"]
    with open(env_path) as f:
        content = f.read()

    missing = []
    for key in required:
        if key not in content:
            missing.append(key)
        else:
            # Check if key has an actual value (not empty, not placeholder)
            for line in content.split("\n"):
                if line.startswith(key + "="):
                    val = line.split("=", 1)[1].strip()
                    if not val or val in ("[REDACTED]", "change-this-to-a-strong-random-key"):
                        missing.append(f"{key} (placeholder/empty)")
                    break

    if missing:
        return False, f"Missing/invalid required env vars: {', '.join(missing)}"
    return True, "OK"


def check_python_deps() -> tuple[bool, str]:
    """Verify core Python dependencies are installed."""
    core_deps = [
        "ollama", "requests", "pydantic", "python_dotenv",
        "psutil", "yaml", "tenacity", "rich", "prompt_toolkit"
    ]
    missing = []
    for dep in core_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        return False, f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt"
    return True, "OK"


def check_ollama_connection() -> tuple[bool, str]:
    """Test connectivity to Ollama server."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import ollama
        client = ollama.Client(host=host)
        models = client.list()
        model_names = [m.get("name", "") for m in models.get("models", [])]
        local_count = sum(1 for m in model_names if ":cloud" not in m)
        cloud_count = sum(1 for m in model_names if ":cloud" in m)
        return True, f"Connected. Models: {local_count} local, {cloud_count} cloud"
    except ImportError:
        return False, "ollama package not installed"
    except Exception as e:
        return False, f"Cannot connect to Ollama at {host}: {e}"


def check_disk_space() -> tuple[bool, str]:
    """Check available disk space."""
    try:
        import psutil
        disk = psutil.disk_usage(str(Path.home()))
        free_gb = disk.free / (1024**3)
        if free_gb < 1:
            return False, f"Low disk space: {free_gb:.1f} GB free"
        return True, f"Disk: {free_gb:.1f} GB free"
    except ImportError:
        return None, "psutil not installed (optional)"
    except Exception:
        return None, "Cannot check disk space"


def check_node_web() -> tuple[bool, str]:
    """Check if web UI dependencies are installed."""
    web_dir = Path(__file__).parent.parent / "web" / "node_modules"
    if web_dir.exists():
        return True, "Web UI node_modules present"
    return None, "Web UI not installed (optional, run: cd web && npm install)"


def check_data_dirs() -> tuple[bool, str]:
    """Verify data directories exist or can be created."""
    data_dir = Path(os.environ.get("AURA_DATA_DIR", Path(__file__).parent.parent / "data"))
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        return True, f"Data dir: {data_dir}"
    except Exception as e:
        return False, f"Cannot create data directory {data_dir}: {e}"


def check_auth_security() -> tuple[bool, str]:
    """Check auth configuration."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return False, ".env not found"

    with open(env_path) as f:
        content = f.read()

    issues = []
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check for live API keys (not redacted)
        if "=" in line:
            key, val = line.split("=", 1)
            val = val.strip()
            if key.endswith("_API_KEY") or key.endswith("_KEY") or key.endswith("_TOKEN"):
                if val and val not in ("[REDACTED]", "", "your_token_here", "your-32-byte-base64-encoded-key") \
                   and not val.startswith("sk-") and not val.startswith("hf_"):
                    if len(val) > 20:
                        issues.append(f"Live key found: {key}={val[:12]}...")

    auth_enabled = False
    for line in content.split("\n"):
        if line.startswith("AURA_API_AUTH_ENABLED=true"):
            auth_enabled = True
            break

    if not auth_enabled:
        issues.append("AURA_API_AUTH_ENABLED is not true")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def run_all_checks() -> dict:
    """Run all health checks, return results dict."""
    checks = {
        "Environment (.env)": check_env_file,
        "Python dependencies": check_python_deps,
        "Ollama connection": check_ollama_connection,
        "Disk space": check_disk_space,
        "Web UI": check_node_web,
        "Data directories": check_data_dirs,
        "Auth security": check_auth_security,
    }

    results = {}
    for name, check_fn in checks.items():
        result = check_fn()
        if result is None:
            continue
        ok, msg = result
        results[name] = (ok, msg)

    return results


def interactive_setup():
    """Interactive setup wizard."""
    print("\n" + "=" * 60)
    print("  AURA Setup Wizard")
    print("=" * 60)
    print()

    # Step 1: .env
    print("[1/4] Checking environment configuration...")
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("  .env not found. Copying from .env.example...")
        example = Path(__file__).parent.parent / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_path)
            print(f"  Created {env_path}")
            print("  Please edit .env and fill in your values:")
            print("  - OLLAMA_API_KEY (from ollama.com/dashboard)")
            print("  - AURA_API_KEY (generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\")")
            print("  - Then run: python scripts/aura_doctor.py again")
            return
        else:
            print("  .env.example also missing! Check your installation.")
            return
    else:
        print("  .env exists")

    # Step 2: Dependencies
    print("\n[2/4] Checking Python dependencies...")
    deps_ok, deps_msg = check_python_deps()
    print(f"  {deps_msg}")
    if not deps_ok:
        print("  Run: pip install -r requirements.txt")
        return

    # Step 3: Ollama
    print("\n[3/4] Checking Ollama connection...")
    ollama_ok, ollama_msg = check_ollama_connection()
    print(f"  {ollama_msg}")
    if not ollama_ok:
        print("  Make sure Ollama is running and OLLAMA_API_KEY is set in .env")
        return

    # Step 4: API key
    print("\n[4/4] Checking API key...")
    key_ok, key_msg = check_auth_security()
    print(f"  {key_msg}")
    if not key_ok:
        print("  See .env.example for configuration guidance")

    print()
    print("=" * 60)
    print("  Setup complete! Run 'python scripts/aura_doctor.py' to verify")
    print("  Start the API: python run_web.py")
    print("  Start the Web UI: cd web && npm run dev")
    print("  Start the CLI: python main.py")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aura Setup Wizard & Health Check")
    parser.add_argument("--setup", action="store_true", help="Interactive setup wizard")
    parser.add_argument("--quick", action="store_true", help="Brief single-line status")
    args = parser.parse_args()

    if args.setup:
        interactive_setup()
        return

    if args.quick:
        results = run_all_checks()
        issues = sum(1 for ok, _ in results.values() if ok is False)
        if issues == 0:
            print("✓ Aura is healthy")
        else:
            print(f"✗ {issues} checks failing. Run 'python scripts/aura_doctor.py' for details")
        return

    # Full check
    print("\n" + "=" * 60)
    print("  Aura Health Check")
    print("=" * 60)
    print()

    results = run_all_checks()
    passed = 0
    failed = 0
    warnings = 0

    for name, (ok, msg) in results.items():
        if ok is True:
            print(f"  ✓ {name}: {msg}")
            passed += 1
        elif ok is False:
            print(f"  ✗ {name}: {msg}")
            failed += 1
        else:
            print(f"  - {name}: {msg}")
            warnings += 1

    print()
    print(f"  {passed} passed, {failed} failed, {warnings} warnings")

    if failed > 0:
        print("\n  Tips:")
        print("  - Missing .env? Copy: cp .env.example .env")
        print("  - Missing packages? Install: pip install -r requirements.txt")
        print("  - Ollama not connecting? Check OLLAMA_HOST and OLLAMA_API_KEY in .env")
        print("  - Run setup wizard: python scripts/aura_doctor.py --setup")

    print()


if __name__ == "__main__":
    main()
