"""
ASGI entrypoint for Vercel serverless deployment.

Vercel's Python runtime automatically looks for an ASGI app instance
in this file or other standard locations.
"""

from api_server import app

# Vercel will automatically use this `app` instance for serverless execution
__all__ = ["app"]
