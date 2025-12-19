# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
    from mcp.shared.message import SessionMessage
except ImportError as e:
    raise ImportError(
        "The 'mcp' package is required to run the MCP server integration but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[mcp]'"
    ) from e


from .tool_context import (
    RawFunctionTool,
    ToolError,
    function_tool,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)

MCPTool = RawFunctionTool


class MCPServer(ABC):
    def __init__(self, *, client_session_timeout_seconds: float) -> None:
        self._client: ClientSession | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._read_timeout = client_session_timeout_seconds

        self._cache_dirty = True
        self._lk_tools: list[MCPTool] | None = None

    @property
    def initialized(self) -> bool:
        return self._client is not None

    def invalidate_cache(self) -> None:
        self._cache_dirty = True

    async def initialize(self) -> None:
        try:
            streams = await self._exit_stack.enter_async_context(self.client_streams())
            receive_stream, send_stream = streams[0], streams[1]
            self._client = await self._exit_stack.enter_async_context(
                ClientSession(
                    receive_stream,
                    send_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout)
                    if self._read_timeout
                    else None,
                )
            )
            await self._client.initialize()  # type: ignore[union-attr]
            self._initialized = True
        except Exception:
            await self.aclose()
            raise

    async def list_tools(self) -> list[MCPTool]:
        if self._client is None:
            raise RuntimeError("MCPServer isn't initialized")

        if not self._cache_dirty and self._lk_tools is not None:
            return self._lk_tools

        tools = await self._client.list_tools()
        lk_tools = [
            self._make_function_tool(tool.name, tool.description, tool.inputSchema, tool.meta)
            for tool in tools.tools
        ]

        self._lk_tools = lk_tools
        self._cache_dirty = False
        return lk_tools

    def _make_function_tool(
        self,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
        meta: dict[str, Any] | None,
    ) -> MCPTool:
        async def _tool_called(raw_arguments: dict[str, Any]) -> Any:
            # In case (somehow), the tool is called after the MCPServer aclose.
            if self._client is None:
                raise ToolError(
                    "Tool invocation failed: internal service is unavailable. "
                    "Please check that the MCPServer is still running."
                )

            tool_result = await self._client.call_tool(name, raw_arguments)

            if tool_result.isError:
                error_str = "\n".join(str(part) for part in tool_result.content)
                raise ToolError(error_str)

            # TODO(theomonnom): handle images & binary messages
            if len(tool_result.content) == 1:
                return tool_result.content[0].model_dump_json()
            elif len(tool_result.content) > 1:
                return json.dumps([item.model_dump() for item in tool_result.content])

            raise ToolError(
                f"Tool '{name}' completed without producing a result. "
                "This might indicate an issue with internal processing."
            )

        raw_schema = {
            "name": name,
            "description": description,
            "parameters": input_schema,
        }
        if meta:
            raw_schema["meta"] = meta

        return function_tool(_tool_called, raw_schema=raw_schema)

    async def aclose(self) -> None:
        try:
            await self._exit_stack.aclose()
        finally:
            self._client = None
            self._lk_tools = None

    @abstractmethod
    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
        | tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback,
        ]
    ]: ...


class MCPServerHTTP(MCPServer):
    """
    HTTP-based MCP server to detect transport type based on URL path.

    - URLs ending with 'sse' use Server-Sent Events (SSE) transport
    - URLs ending with 'mcp' use streamable HTTP transport
    - For other URLs, defaults to SSE transport for backward compatibility

    Note: SSE transport is being deprecated in favor of streamable HTTP transport.
    See: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.url = url
        self.headers = headers
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout
        self._use_streamable_http = self._should_use_streamable_http(url)

    def _should_use_streamable_http(self, url: str) -> bool:
        """
        Determine transport type based on URL path.

        Returns True for streamable HTTP if URL ends with 'mcp',
        False for SSE if URL ends with 'sse' or for backward compatibility.
        """
        parsed_url = urlparse(url)
        path_lower = parsed_url.path.lower().rstrip("/")
        return True

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
        | tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback,
        ]
    ]:
        if self._use_streamable_http:
            return streamablehttp_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=timedelta(seconds=self._timeout),
                sse_read_timeout=timedelta(seconds=self._sse_read_timeout),
            )
        else:
            return sse_client(  # type: ignore[no-any-return]
                url=self.url,
                headers=self.headers,
                timeout=self._timeout,
                sse_read_timeout=self._sse_read_timeout,
            )

    def __repr__(self) -> str:
        transport_type = "streamable_http" if self._use_streamable_http else "sse"
        return f"MCPServerHTTP(url={self.url}, transport={transport_type})"


class MCPServerHTTPFiltered(MCPServerHTTP):
    """
    HTTP-based MCP server with tool filtering capability.

    Only tools whose names are in the `allowed_tools` list will be provided to the agent.
    If `allowed_tools` is `None` or empty, all tools from the server will be available
    (same behavior as `MCPServerHTTP`).

    This class extends `MCPServerHTTP` with the same transport detection behavior:
    - URLs ending with 'sse' use Server-Sent Events (SSE) transport
    - URLs ending with 'mcp' use streamable HTTP transport
    - For other URLs, defaults to SSE transport for backward compatibility
    """

    def __init__(
        self,
        url: str,
        allowed_tools: list[str] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(
            url=url,
            headers=headers,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
            client_session_timeout_seconds=client_session_timeout_seconds,
        )
        self._allowed_tools = set(allowed_tools) if allowed_tools else None

    async def list_tools(self) -> list[MCPTool]:
        """
        List tools from the MCP server, filtered by allowed_tools if specified.
        """
        all_tools = await super().list_tools()

        # If no filter is set, return all tools
        if self._allowed_tools is None:
            return all_tools

        # Filter tools by name
        filtered_tools = []
        for tool in all_tools:
            # Get tool name based on tool type
            if is_function_tool(tool):
                tool_name = get_function_info(tool).name
            elif is_raw_function_tool(tool):
                tool_name = get_raw_function_info(tool).name
            else:
                # Fallback: try to get name from raw_schema if available
                # This shouldn't happen for MCP tools, but handle it gracefully
                continue

            if tool_name in self._allowed_tools:
                filtered_tools.append(tool)

        return filtered_tools

    def __repr__(self) -> str:
        transport_type = "streamable_http" if self._use_streamable_http else "sse"
        allowed_str = (
            f", allowed_tools={list(self._allowed_tools)}"
            if self._allowed_tools
            else ""
        )
        return f"MCPServerHTTPFiltered(url={self.url}, transport={transport_type}{allowed_str})"


class MCPServerStdio(MCPServer):
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        return stdio_client(  # type: ignore[no-any-return]
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"


async def create_mcp_tools(
    url: str,
    tool_names: list[str],
    headers: dict[str, Any] | None = None,
    timeout: float = 5,
    sse_read_timeout: float = 60 * 5,
    client_session_timeout_seconds: float = 5,
) -> tuple[list[MCPTool], MCPServerHTTPFiltered]:
    """
    Create individual MCP tools from an HTTP MCP server that can be attached to agents.

    This function connects to an MCP server, fetches the specified tools, and returns
    them as a list that can be passed to an Agent's `tools` parameter. The tools are
    filtered to only include those specified in `tool_names`.

    **Important**: The returned `MCPServerHTTPFiltered` instance must be kept alive
    for the tools to work. The tools reference the server's client connection internally.
    You can store the server instance as an attribute of your agent or in a variable
    that persists for the lifetime of the agent.

    Args:
        url: The URL of the MCP server (e.g., "http://localhost:8000/mcp")
        tool_names: List of tool names to fetch from the MCP server
        headers: Optional HTTP headers to include in requests
        timeout: Connection timeout in seconds (default: 5)
        sse_read_timeout: SSE read timeout in seconds (default: 300)
        client_session_timeout_seconds: Client session timeout in seconds (default: 5)

    Returns:
        A tuple of (list of tools, MCPServerHTTPFiltered instance).
        The server instance must be kept alive for the tools to function.

    Example:
        ```python
        from livekit.agents import Agent, mcp

        # Create tools from MCP server
        tools, mcp_server = await mcp.create_mcp_tools(
            url="http://localhost:8000/mcp",
            tool_names=["tool1", "tool2"],
            headers={"Authorization": "Bearer token"}
        )

        # Create agent with the tools
        agent = Agent(
            instructions="You are a helpful assistant",
            tools=tools  # Tools are attached at agent level
        )

        # Store the server to keep it alive
        agent._mcp_server = mcp_server
        ```
    """
    server = MCPServerHTTPFiltered(
        url=url,
        allowed_tools=tool_names,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=sse_read_timeout,
        client_session_timeout_seconds=client_session_timeout_seconds,
    )

    # Initialize the server connection
    await server.initialize()

    # Get the filtered tools
    tools = await server.list_tools()

    return tools, server
