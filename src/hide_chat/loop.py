"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined tools.
"""

import json
import platform
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from hide_chat.tools.collection import RemoteToolCollection

from .tools import BashTool, EditTool, ToolCollection, ToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"
MODEL_NAME = "claude-3-5-sonnet-20241022"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising a Mac using {platform.machine()} architecture with internet access.
* You can feel free to install Mac applications with your bash tool. Use curl instead of wget.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>"""


async def sampling_loop(
    *,
    model: str = MODEL_NAME,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    max_tokens: int = 4096,
    mcp_url: str | None = None,
):
    """
    Agentic sampling loop for the assistant/tool interaction.
    """
    if mcp_url:
        tool_collection = RemoteToolCollection(url=mcp_url)
    else:
        tool_collection = ToolCollection(
            BashTool(),
            EditTool(),
        )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        client = Anthropic(api_key=api_key, max_retries=16)
        betas = [PROMPT_CACHING_BETA_FLAG, COMPUTER_USE_BETA_FLAG]
        # _inject_prompt_caching(messages)
        system["cache_control"] = {"type": "ephemeral"}

        try:
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=await tool_collection.to_params(),
                betas=betas,
                stream=True,
            )

            # Store the request for later
            request = raw_response.http_response.request
            accumulated_content = []
            response_params = []
            current_block = None

            for chunk in raw_response.parse():
                accumulated_content.append(
                    chunk.model_dump_json()
                )  # Accumulate the content
                if chunk.type == "message_start":
                    continue
                elif chunk.type == "content_block_start":
                    current_block = {"type": chunk.content_block.type}
                    if chunk.content_block.type == "tool_use":
                        current_block.update(
                            {
                                "name": chunk.content_block.name,
                                "id": chunk.content_block.id,
                                "input": "",
                            }
                        )
                elif chunk.type == "content_block_delta":
                    if current_block["type"] == "text":
                        if "text" not in current_block:
                            current_block["text"] = ""
                        current_block["text"] += chunk.delta.text
                        output_callback(current_block)
                    elif current_block["type"] == "tool_use":
                        if chunk.delta.partial_json:
                            current_block["input"] += chunk.delta.partial_json
                        output_callback(current_block)
                elif chunk.type == "content_block_stop":
                    if current_block:
                        if current_block["type"] == "tool_use":
                            current_block["input"] = json.loads(current_block["input"])
                        response_params.append(current_block)
                        current_block = None

            # Create a response with the accumulated content
            response = httpx.Response(
                status_code=raw_response.http_response.status_code,
                headers=raw_response.http_response.headers,
                content=f'[{",".join(accumulated_content)}]'.encode(),
                request=request,
            )
            api_response_callback(request, response, None)

            # Add the assistant's response to messages
            if response_params:
                messages.append({"role": "assistant", "content": response_params})

            tool_result_content: list[BetaToolResultBlockParam] = []
            for content_block in response_params:
                if content_block["type"] == "tool_use":
                    result = await tool_collection.run(
                        name=content_block["name"],
                        tool_input=cast(dict[str, Any], content_block["input"]),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block["id"])
                    )
                    tool_output_callback(result, content_block["id"])

            if tool_result_content:
                messages.append({"content": tool_result_content, "role": "user"})
                continue

            return

        except (APIStatusError, APIResponseValidationError) as e:
            api_response_callback(e.request, e.response, e)
            return
        except APIError as e:
            api_response_callback(e.request, e.body, e)
            return


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
