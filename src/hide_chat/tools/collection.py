"""Collection classes for managing multiple tools."""

import logging
from typing import Any

from anthropic.types.beta import BetaToolParam, BetaToolUnionParam
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
import mcp.types as types

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)

logger = logging.getLogger(__name__)

class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool):
        self.tools = tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in tools}

    async def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)


class RemoteToolCollection:
    """A collection of tools provided by a MCP server."""

    def __init__(self, *, url: str):
        self.url = url

    async def to_params(self) -> list[BetaToolParam]:
        async with sse_client(self.url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                result = await session.list_tools()
                return [
                    BetaToolParam(
                        name=tool.name,
                        input_schema=await self.__remove_fields(tool.inputSchema),
                        description=tool.description or "",
                    )
                    for tool in result.tools
                ]

    # TODO: fix this
    async def __remove_fields(self, schema: dict[str, Any]) -> dict[str, Any]:
        if "oneOf" in schema:
            del schema["oneOf"]
        return schema

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        async with sse_client(self.url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()

                result = await session.call_tool(name, tool_input)
                if result.isError:
                    if len(result.content) == 0:
                        return ToolFailure(error="Tool returned an error but no content")
                    return ToolFailure(
                        error=". ".join(
                            content.text
                            for content in result.content
                            if isinstance(content, types.TextContent)
                        )
                    )

                if len(result.content) > 1:
                    logger.warning(
                        "Tool %s returned multiple results: %s. Ignoring all but the first.",
                        name,
                        result.content,
                    )

                if not isinstance(result.content[0], types.TextContent):
                    logger.warning(
                        "Tool %s returned a non-text result: %s. Not supported.",
                        name,
                        result.content[0],
                    )
                    return ToolResult(error="Tool returned a non-text result")

                return ToolResult(output=result.content[0].text)
