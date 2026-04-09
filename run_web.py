#!/usr/bin/env python3
"""
AURA Web UI Launcher

Launches the FastAPI backend server for the modern web interface.

Usage:
    python run_web.py              # Start API server only (dev mode)
    python run_web.py --prod       # Production mode (serve built frontend)
    python run_web.py --port 8080  # Custom port

For development:
    1. Run this script: python run_web.py
    2. In another terminal: cd web && npm run dev
    3. Open http://localhost:5173
"""

import os
import sys
import argparse
import signal

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import fastapi
    except ImportError:
        missing.append('fastapi')

    try:
        import uvicorn
    except ImportError:
        missing.append('uvicorn')

    try:
        import websockets
    except ImportError:
        missing.append('websockets')

    if missing:
        print("Missing dependencies. Please install:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def start_api_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    import uvicorn

    print(f"""
    +---------------------------------------------------------------+
    |                      AURA Web Interface                       |
    +---------------------------------------------------------------+
    |  API Server: http://{host}:{port}                            |
    |  API Docs:   http://{host}:{port}/docs                       |
    |                                                               |
    |  For development, also run:                                   |
    |    cd web && npm install && npm run dev                       |
    |  Then open: http://localhost:5173                             |
    +---------------------------------------------------------------+
    """)

    # Use full module path for uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[os.path.join(PROJECT_ROOT, 'api')] if reload else None,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(
        description="AURA Web UI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web.py                 Start dev server on port 8000
  python run_web.py --port 8080     Use custom port
  python run_web.py --prod          Production mode (no reload)
  python run_web.py --host 0.0.0.0  Allow external connections
        """
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Production mode (disable reload)"
    )

    args = parser.parse_args()

    # Set AURA_ENV for production mode before importing app
    if args.prod:
        os.environ["AURA_ENV"] = "production"

    # Check dependencies
    check_dependencies()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nShutting down AURA Web Server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start server
    start_api_server(
        host=args.host,
        port=args.port,
        reload=not args.prod
    )


if __name__ == "__main__":
    main()
