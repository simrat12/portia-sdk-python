"""MCP Session Tests."""

import subprocess
from pathlib import Path

import mcp
import pytest

from portia.mcp_session import SseMcpClientConfig, StdioMcpClientConfig, get_mcp_session

SERVER_FILE_PATH = Path(__file__).parent / "mcp_server.py"


@pytest.mark.asyncio
async def test_mcp_session_stdio() -> None:
    """Test the MCP session with stdio."""
    async with get_mcp_session(
        StdioMcpClientConfig(
            server_name="test_server",
            command="poetry",
            args=["run", "python", str(SERVER_FILE_PATH.absolute()), "stdio"],
        ),
    ) as session:
        tools = await session.list_tools()
        assert isinstance(tools, mcp.ListToolsResult)
        assert len(tools.tools) == 1


@pytest.mark.asyncio
async def test_mcp_session_sse() -> None:
    """Test the MCP session with SSE."""
    process = subprocess.Popen(  # noqa: ASYNC220, S602
        ["poetry", "run", "python", str(SERVER_FILE_PATH.absolute()), "sse"],  # noqa: S607
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with process:
        async with get_mcp_session(
            SseMcpClientConfig(
                server_name="test_server",
                url="http://localhost:11385/sse",
                sse_read_timeout=5,
                timeout=5,
            ),
        ) as session:
            tools = await session.list_tools()
            assert isinstance(tools, mcp.ListToolsResult)
            assert len(tools.tools) == 1
