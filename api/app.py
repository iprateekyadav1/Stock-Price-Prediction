"""FastAPI entrypoint for Vercel.

Vercel's Python runtime scans for an ASGI `app` object in
common locations like `api/app.py`. This thin module simply
re-exports the main FastAPI instance from `api_server.py`.
"""

from api_server import app  # noqa: F401
