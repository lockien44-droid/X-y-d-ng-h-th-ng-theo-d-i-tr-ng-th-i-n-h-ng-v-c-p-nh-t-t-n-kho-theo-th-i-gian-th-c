"""Render compatibility entrypoint.

Some Render services are still configured with:

    gunicorn ... websocket_server:app

The current application lives in app/server.py as app.server:app.
This small shim keeps the old Render start command working.
"""

from app.server import app, socketio  # noqa: F401

