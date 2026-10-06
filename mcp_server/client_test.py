"""
client_test.py - Official MCP Client protocol verification test suite.
Spawns the MCP server subprocess via stdio, performs full protocol handshake,
discovers all tools, resources, and prompts, and executes verification tests.
"""

import sys
import asyncio
import logging
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_client_test")


async def run_protocol_tests():
    python_exe = sys.executable
    project_root = str(Path(__file__).parent.parent.resolve())

    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_server.server"],
        cwd=project_root,
    )

    print("\n" + "=" * 60)
    print("MCP CLIENT PROTOCOL VERIFICATION TEST")
    print("=" * 60)
    print(f"[1/6] Launching MCP Server via stdio: {python_exe} -m mcp_server.server")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize session
            print("[2/6] Initializing MCP ClientSession Handshake...")
            init_result = await session.initialize()
            print(f"  [OK] Connected to: {init_result.server_info.name} (v{init_result.server_info.version})")

            # 2. List Tools
            print("\n[3/6] Discovering MCP Tools...")
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"  [OK] Found {len(tool_names)} tools: {tool_names}")
            expected_tools = [
                "query_reviews_rag",
                "semantic_search_reviews",
                "triage_customer_feedback",
                "get_triage_batch_metrics",
            ]
            for t_name in expected_tools:
                assert t_name in tool_names, f"Missing tool: {t_name}"

            # 3. List Resources
            print("\n[4/6] Discovering MCP Resources...")
            resources_result = await session.list_resources()
            resource_uris = [str(r.uri) for r in resources_result.resources]
            print(f"  [OK] Found {len(resource_uris)} resources: {resource_uris}")
            expected_resources = ["reviews://dataset-summary", "triage://queues"]
            for r_uri in expected_resources:
                assert r_uri in resource_uris, f"Missing resource: {r_uri}"

            # 4. List Prompts
            print("\n[5/6] Discovering MCP Prompts...")
            prompts_result = await session.list_prompts()
            prompt_names = [p.name for p in prompts_result.prompts]
            print(f"  [OK] Found {len(prompt_names)} prompts: {prompt_names}")
            assert "feedback_triage_prompt" in prompt_names, "Missing prompt: feedback_triage_prompt"

            # 5. Execute Tool Test (get_triage_batch_metrics)
            print("\n[6/6] Executing MCP Tool Call via JSON-RPC protocol...")
            tool_call_result = await session.call_tool("get_triage_batch_metrics", arguments={})
            print("  [OK] Tool execution result received:")
            for content_item in tool_call_result.content:
                if hasattr(content_item, "text"):
                    print(f"    Payload: {content_item.text[:200]}...")

            # 6. Read Resource Test
            print("\n[+] Reading Resource: reviews://dataset-summary...")
            res_content = await session.read_resource("reviews://dataset-summary")
            print(f"  [OK] Resource content items: {len(res_content.contents)}")

            print("\n" + "=" * 60)
            print("ALL MCP PROTOCOL VERIFICATION TESTS PASSED SUCCESSFULLY! (6/6)")
            print("=" * 60 + "\n")


def main():
    asyncio.run(run_protocol_tests())


if __name__ == "__main__":
    main()
