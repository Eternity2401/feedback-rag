"""
http_server.py - Public Streamable HTTP ASGI deployment for Render Free Web Service.
Exposes /mcp (Streamable HTTP) and /healthz (Lightweight Health Check).
"""

import os
import sys
import logging
from typing import List
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.applications import Starlette

from mcp.server.transport_security import TransportSecuritySettings
from mcp_server.server import mcp_server

logger = logging.getLogger("mcp_server.http_server")

# Dynamic Host and Port Configuration for Local and Cloud (Render) Environments
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
RENDER_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME", "")

# Build dynamic allowed hosts list
allowed_hosts: List[str] = ["localhost", "127.0.0.1", "0.0.0.0", "*"]
if RENDER_HOSTNAME:
    allowed_hosts.append(RENDER_HOSTNAME)
    allowed_hosts.append(f"*.{RENDER_HOSTNAME}")

transport_security = TransportSecuritySettings(
    allowed_hosts=allowed_hosts,
    allowed_origins=["*"],
    enable_dns_rebinding_protection=False if not RENDER_HOSTNAME else True,
)

# Create the Streamable HTTP Starlette Application
app: Starlette = mcp_server.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    transport_security=transport_security,
    host=HOST,
)


# Add Health Check endpoint for Render deployment monitoring (Fast, 0 Gemini quota)
async def healthz(request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "feedback-rag-mcp",
            "protocol": "Model Context Protocol (v2)",
            "transport": "Streamable HTTP (/mcp)",
            "tools_count": 4,
            "resources_count": 2,
            "prompts_count": 1,
        }
    )


# Attach /healthz route
app.routes.append(Route("/healthz", healthz, methods=["GET"]))


def main():
    """Runs the HTTP server using uvicorn."""
    import uvicorn

    logger.info(f"Starting feedback-rag Public MCP Server on http://{HOST}:{PORT}/mcp")
    uvicorn.run(
        "mcp_server.http_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
