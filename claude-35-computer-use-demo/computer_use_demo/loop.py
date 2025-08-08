"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
import time
from collections.abc import Callable
import sys
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
# Ensure prints flush line-by-line even when not attached to a TTY or when piped
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
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

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "arn:aws:bedrock:us-west-2:302263051492:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    max_actions: int = 100,
    rate_limit_delay: float = 2.0,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting Claude computer use session")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model: {model}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Provider: {provider}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Max actions: {max_actions}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rate limit delay: {rate_limit_delay} seconds")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Max tokens: {max_tokens}")
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    # Add initial delay to ensure we don't hit rate limits on first call
    if rate_limit_delay > 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial delay of {rate_limit_delay} seconds to avoid rate limiting...")
        time.sleep(rate_limit_delay)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial delay completed. Starting main loop.")
    
    for action_num in range(max_actions):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] === Starting action {action_num + 1}/{max_actions} ===")
        enable_prompt_caching = False
        betas = [COMPUTER_USE_BETA_FLAG]
        image_truncation_threshold = 10
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Is it ever worth it to bust the cache with prompt caching?
            image_truncation_threshold = 50
            system["cache_control"] = {"type": "ephemeral"}

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        # Add delay BEFORE API call to ensure rate limiting compliance
        if rate_limit_delay > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {rate_limit_delay} seconds before API call to avoid rate limiting...")
            time.sleep(rate_limit_delay)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Pre-API delay completed. Making API call now...")
        
        # Call the API with retry logic for rate limiting
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        max_retries = 3
        retry_delay = rate_limit_delay
        
        for attempt in range(max_retries):
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Making API call attempt {attempt + 1}/{max_retries}...")
                api_call_start = time.time()
                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                )
                api_call_duration = time.time() - api_call_start
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ API call successful! Duration: {api_call_duration:.2f} seconds")
                break  # Success, exit retry loop
            except APIStatusError as e:
                if e.status_code == 429:  # Rate limit error
                    if attempt < max_retries - 1:
                        retry_wait = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Rate limit hit (429)! Retrying in {retry_wait} seconds... (attempt {attempt + 1}/{max_retries})")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rate limit error details: {e}")
                        time.sleep(retry_wait)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Retry delay completed. Attempting API call again...")
                        continue
                # If not rate limit or max retries reached, handle as before
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ API error (not rate limit): {e}")
                api_response_callback(e.request, e.response, e)
                return messages
            except (APIResponseValidationError) as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ API validation error: {e}")
                api_response_callback(e.request, e.response, e)
                return messages
            except APIError as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ General API error: {e}")
                api_response_callback(e.request, e.body, e)
                return messages

        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing API response...")
        response = raw_response.parse()

        response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Response parsed. Processing {len(response_params)} content blocks...")

        tool_result_content: list[BetaToolResultBlockParam] = []
        tool_use_count = 0
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                tool_use_count += 1
                tool_name = content_block["name"]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing tool {tool_use_count}: {tool_name}")
                tool_start = time.time()
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )
                tool_duration = time.time() - tool_start
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Tool {tool_name} completed in {tool_duration:.2f} seconds")
                tool_result_content.append(
                    _make_api_tool_result(result, content_block["id"])
                )
                tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No tool use detected. Task completed.")
            return messages

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding {len(tool_result_content)} tool results to conversation. Continuing to next iteration...")
        messages.append({"content": tool_result_content, "role": "user"})
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Exhausted {max_actions} actions limit")
    return messages


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


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
