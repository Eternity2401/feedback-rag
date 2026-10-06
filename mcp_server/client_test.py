"""
client_test.py - Official MCP Client protocol verification test suite.
Supports dual-mode verification:
1. Local stdio transport test
2. Remote Streamable HTTP (Render) transport test
"""

import sys
import asyncio
import logging
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_client_test")

REMOTE_MCP_URL = "https://feedback-rag-mcp.onrender.com/mcp"


async def run_stdio_tests():
    python_exe = sys.executable
    project_root = str(Path(__file__).parent.parent.resolve())

    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_server.server"],
        cwd=project_root,
    )

    print("\n" + "=" * 60)
    print("MCP CLIENT PROTOCOL TEST: LOCAL STDIO TRANSPORT")
    print("=" * 60)
    print(f"[1/6] Launching MCP Server via stdio: {python_exe} -m mcp_server.server")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            print("[2/6] Initializing MCP ClientSession Handshake...")
            init_result = await session.initialize()
            print(f"  [OK] Connected to: {init_result.server_info.name} (v{init_result.server_info.version})")

            print("\n[3/6] Discovering MCP Tools...")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"  [OK] Found {len(tool_names)} tools: {tool_names}")

            print("\n[4/6] Discovering MCP Resources...")
            resources_result = await session.list_resources()
            resource_uris = [str(r.uri) for r in resources_result.resources]
            print(f"  [OK] Found {len(resource_uris)} resources: {resource_uris}")

            print("\n[5/6] Discovering MCP Prompts...")
            prompts_result = await session.list_prompts()
            prompt_names = [p.name for p in prompts_result.prompts]
            print(f"  [OK] Found {len(prompt_names)} prompts: {prompt_names}")

            print("\n[6/6] Executing MCP Tool Call (get_triage_batch_metrics)...")
            tool_call_result = await session.call_tool("get_triage_batch_metrics", arguments={})
            print("  [OK] Tool execution result received:")
            for content_item in tool_call_result.content:
                if hasattr(content_item, "text"):
                    print(f"    Payload: {content_item.text[:150]}...")

            print("\n" + "=" * 60)
            print("LOCAL STDIO MCP TESTS PASSED SUCCESSFULLY! (6/6)")
            print("=" * 60)


async def run_remote_tests(url: str = REMOTE_MCP_URL):
    print("\n" + "=" * 60)
    print("MCP CLIENT PROTOCOL TEST: REMOTE STREAMABLE HTTP TRANSPORT")
    print(f"Target URL: {url}")
    print("=" * 60)

    try:
        async with streamable_http_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                print("[1/5] Initializing Remote MCP ClientSession Handshake...")
                init_result = await session.initialize()
                print(f"  [OK] Remote Connected to: {init_result.server_info.name} (v{init_result.server_info.version})")

                print("\n[2/5] Discovering Remote Tools...")
                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                print(f"  [OK] Found {len(tool_names)} tools: {tool_names}")

                print("\n[3/5] Discovering Remote Resources...")
                resources_result = await session.list_resources()
                resource_uris = [str(r.uri) for r in resources_result.resources]
                print(f"  [OK] Found {len(resource_uris)} resources: {resource_uris}")

                print("\n[4/5] Discovering Remote Prompts...")
                prompts_result = await session.list_prompts()
                prompt_names = [p.name for p in prompts_result.prompts]
                print(f"  [OK] Found {len(prompt_names)} prompts: {prompt_names}")

                print("\n[5/5] Executing Remote Tool Call (get_triage_batch_metrics)...")
                tool_call_result = await session.call_tool("get_triage_batch_metrics", arguments={})
                print("  [OK] Remote tool execution payload received:")
                for content_item in tool_call_result.content:
                    if hasattr(content_item, "text"):
                        print(f"    Payload: {content_item.text[:150]}...")

                print("\n" + "=" * 60)
                print("REMOTE STREAMABLE HTTP MCP TESTS PASSED SUCCESSFULLY! (5/5)")
                print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Remote MCP test failed: {e}")
        print("Note: If Render service is sleeping (cold start), wait 30 seconds and retry.")


def main():
    if "--remote" in sys.argv:
        asyncio.run(run_remote_tests())
    elif "--stdio" in sys.argv:
        asyncio.run(run_stdio_tests())
    else:
        # Run both by default
        asyncio.run(run_stdio_tests())
        asyncio.run(run_remote_tests())


if __name__ == "__main__":
    main()
