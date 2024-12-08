"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast
import json

from dotenv import load_dotenv
import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from hide_chat.loop import sampling_loop
from hide_chat.tools import ToolResult
from hide_mcp.sandbox import create_sandbox, kill_sandbox, setup_hide_mcp

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button with a friendly blue */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }

    /* Style for thread buttons */
    .thread-button {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: white;
        transition: all 0.2s ease;
    }
    .thread-button:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    
    /* Style for action buttons */
    .action-button {
        border-radius: 20px;
        padding: 8px 16px;
        margin: 8px 0;
        width: 100%;
    }
    
    /* Thread preview text styling */
    .thread-preview {
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.4;
        max-width: 100%;
        padding: 8px 0;
        text-align: left;
    }
    
    /* Make buttons more compact and left-aligned */
    .stButton button {
        padding: 0.5rem 1rem;
        line-height: 1.4;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    
    /* Override Streamlit's default button centering */
    .stButton {
        text-align: left !important;
    }
    
    /* Hide default Streamlit margins in sidebar */
    .block-container {
        padding-top: 1rem;
    }
</style>
"""

INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None
    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = load_chat_threads()
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False
    if "sandbox" not in st.session_state:
        st.session_state.sandbox = None
    if "mcp_url" not in st.session_state:
        st.session_state.mcp_url = None


async def main():
    """Render loop for streamlit"""
    load_dotenv()
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Hide Chat")

    with st.sidebar:
        if st.button("New Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_thread_id = None
            if st.session_state.sandbox:
                kill_sandbox(st.session_state.sandbox)
            st.session_state.sandbox = None
            st.session_state.mcp_url = None
            st.rerun()

        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        # List existing threads with better formatting
        for thread_id, thread_data in sorted(
            st.session_state.chat_threads.items(),
            key=lambda x: x[1]["last_updated"],
            reverse=True
        ):
            timestamp = datetime.fromisoformat(thread_data["last_updated"]).strftime("%b %d, %H:%M")

            with st.container():
                preview_text = thread_data["preview"]
                timestamp_text = f":clock2: {timestamp}"

                if st.button(
                    f"{preview_text}\n\n{timestamp_text}",
                    key=f"thread_{thread_id}",
                    use_container_width=True,
                ):
                    if hasattr(st.session_state, 'current_sender'):
                        del st.session_state.current_sender
                    if hasattr(st.session_state, 'current_type'):
                        del st.session_state.current_type
                    st.session_state.current_thread_id = thread_id
                    st.session_state.messages = thread_data["messages"]
                    if st.session_state.sandbox:
                        kill_sandbox(st.session_state.sandbox)
                    st.session_state.sandbox = None
                    st.session_state.mcp_url = None
                    st.rerun()

        # Place Reset button at the bottom
        st.markdown(
            """
            <div style='position: fixed; bottom: 20px; width: inherit;'>
                <hr style='margin: 0 -1rem 1rem -1rem'>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Reset All", type="secondary", use_container_width=True):
            with st.spinner("Resetting..."):
                st.session_state.clear()
                setup_state()

    if not st.session_state.auth_validated:
        if auth_error := validate_auth(st.session_state.api_key):
            st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")
            return
        else:
            st.session_state.auth_validated = True

    if not st.session_state.sandbox:
        with st.spinner("Setting up sandbox..."):
            sbx = create_sandbox()
            st.session_state.sandbox = sbx
            st.toast("Successfully created sandbox", icon="🖥️")

        with st.spinner("Installing MCP..."):
            st.session_state.mcp_url = setup_hide_mcp(sbx)
            st.toast("Successfully installed MCP", icon="🔗")
            
        st.toast("All set up!", icon="🚀")

    chat, http_logs = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input("Type a message to send to Hide...")

    with chat:
        # render past chats
        for message in st.session_state.messages:
            if isinstance(message["content"], str):
                _render_message(message["role"], message["content"])
            elif isinstance(message["content"], list):
                for block in message["content"]:
                    _render_message(
                        Sender.TOOL if block["type"] == "tool_result" else message["role"],
                        cast(BetaContentBlockParam | ToolResult, block),
                    )

        # render past http exchanges
        for identity, (request, response) in st.session_state.responses.items():
            _render_api_response(request, response, identity, http_logs)

        # render past chats
        if new_message:
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": [
                        # *maybe_add_interruption_blocks(),
                        BetaTextBlockParam(type="text", text=new_message),
                    ],
                }
            )
            _render_message(Sender.USER, new_message)
            # Auto-save after new message
            st.session_state.current_thread_id = save_chat_thread(
                st.session_state.messages,
                st.session_state.current_thread_id
            )

        try:
            most_recent_message = st.session_state["messages"][-1]
        except IndexError:
            return

        if most_recent_message["role"] is not Sender.USER:
            # we don't have a user message to respond to, exit early
            return

        with track_sampling_loop():
            # run the agent sampling loop with the newest message
            await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=http_logs,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                mcp_url=st.session_state.mcp_url,
            )
            # Save chat thread after bot response
            st.session_state.current_thread_id = save_chat_thread(
                st.session_state.messages,
                st.session_state.current_thread_id
            )


def maybe_add_interruption_blocks():
    if not st.session_state.in_sampling_loop:
        return []
    # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
    # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
    result = []
    last_message = st.session_state.messages[-1]
    previous_tool_use_ids = [
        block["id"] for block in last_message["content"] if block["type"] == "tool_use"
    ]
    for tool_use_id in previous_tool_use_ids:
        st.session_state.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
        result.append(
            BetaToolResultBlockParam(
                tool_use_id=tool_use_id,
                type="tool_result",
                content=INTERRUPT_TOOL_ERROR,
                is_error=True,
            )
        )
    result.append(BetaTextBlockParam(type="text", text=INTERRUPT_TEXT))
    return result


@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False


def validate_auth(api_key: str | None):
    if not api_key:
        return "Enter your Anthropic API key in the sidebar to continue."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error)
    _render_api_response(request, response, response_id, tab)


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and rendering it."""
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)


def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: Sender,
    message: str | BetaContentBlockParam | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    is_tool_result = not isinstance(message, str | dict)
    if not message:
        return
    
    # Get message type
    message_type = "tool_result" if is_tool_result else (
        message["type"] if isinstance(message, dict) else "text"
    )
    
    # Create new message container if sender or type changes
    if (not hasattr(st.session_state, 'current_sender') or 
        st.session_state.current_sender != sender or
        st.session_state.current_type != message_type):
        st.session_state.current_message = st.chat_message(sender)
        if message_type in ["tool_use", "tool_result"]:
            with st.session_state.current_message.expander(message_type.replace("_", " ").title(), expanded=False):
                st.session_state.current_placeholder = st.empty()
        else:
            st.session_state.current_placeholder = st.session_state.current_message.empty()
        st.session_state.current_sender = sender
        st.session_state.current_type = message_type

    if is_tool_result:
        message = cast(ToolResult, message)
        if message.output:
            if message.__class__.__name__ == "CLIResult":
                st.session_state.current_placeholder.code(message.output)
            else:
                st.session_state.current_placeholder.markdown(message.output)
        if message.error:
            st.session_state.current_placeholder.error(message.error)
        if message.base64_image:
            st.session_state.current_placeholder.image(base64.b64decode(message.base64_image))
        st.session_state.current_sender = None
    elif isinstance(message, dict):
        if message["type"] == "text":
            st.session_state.current_placeholder.markdown(message["text"])
        elif message["type"] == "tool_use":
            st.session_state.current_placeholder.code(
                f'Tool: {message["name"]}\nInput: {message["input"]}'
            )
        elif message["type"] == "tool_result":
            for content in message.get("content", []):
                if content.get("type") == "text":
                    st.session_state.current_placeholder.code(content.get("text"))
        else:
            raise Exception(f'Unexpected response type {message["type"]}')
    else:
        st.session_state.current_placeholder.markdown(message)


def save_chat_thread(messages, thread_id=None):
    """Save chat thread to storage"""
    if not thread_id:
        thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    thread_data = {
        "messages": messages,
        "last_updated": datetime.now().isoformat(),
        "preview": _generate_preview(messages)
    }
    
    try:
        threads_dir = CONFIG_DIR / "chat_threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        
        with open(threads_dir / f"{thread_id}.json", "w") as f:
            json.dump(thread_data, f)
        return thread_id
    except Exception as e:
        st.error(f"Failed to save chat thread: {e}")
        return None

def load_chat_threads():
    """Load all chat threads from storage"""
    threads_dir = CONFIG_DIR / "chat_threads"
    threads = {}

    if threads_dir.exists():
        for thread_file in threads_dir.glob("*.json"):
            try:
                with open(thread_file, "r") as f:
                    thread_data = json.load(f)
                    thread_id = thread_file.stem
                    threads[thread_id] = thread_data
            except Exception as e:
                st.error(f"Failed to load thread {thread_file}: {e}")

    return threads


def _generate_preview(messages, max_length=100):
    """Generate a preview of the chat thread from the first user message"""
    for message in messages:
        if message["role"] == Sender.USER:
            content = message["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block["type"] == "text":
                        text = block["text"].replace("\n", " ")
                        return text[:max_length] + ("..." if len(text) > max_length else "")
            elif isinstance(content, str):
                text = content.replace("\n", " ")
                return text[:max_length] + ("..." if len(text) > max_length else "")
    return "New conversation"


if __name__ == "__main__":
    asyncio.run(main())
