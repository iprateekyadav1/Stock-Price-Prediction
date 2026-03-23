"""FastAPI entrypoint for Vercel.

Vercel's Python runtime scans for an ASGI `app` object in
common locations like `api/app.py`. This thin module ensures the
repository root is on sys.path so `api_server` can be imported,
and re-exports the main FastAPI instance from `api_server.py`.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path so imports like `import api_server` work
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_server import app  # noqa: F401
