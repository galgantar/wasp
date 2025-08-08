# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
from enum import StrEnum
import json
import os

import click
from computer_use_demo.loop import sampling_loop, APIProvider
from anthropic.types.beta import BetaContentBlockParam, BetaTextBlockParam
from computer_use_demo.tools import ToolResult
import httpx


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def _output_callback(message: BetaContentBlockParam):
    pass


def _tool_callback(message: ToolResult, some_other_str: str):
    pass


class APIRequestsLogger:
    def __init__(self, file_path_to_write_to: str):
        self.file_path_to_write_to = file_path_to_write_to
        self.turn = 0

    def log_claude(
        self,
        request: httpx.Request,
        response: httpx.Response | object | None,
        error: Exception | None,
    ):
        with open(self.file_path_to_write_to, "a") as f:
            latest_request = json.loads(request.content.decode("utf-8"))

            latest_messages = latest_request["messages"]
            system_message_content = latest_request["system"]
            system_message = {"role": "system", "content": system_message_content}
            latest_messages.insert(0, system_message)
            f.write(json.dumps(latest_messages) + "\n")

            if not error:
                latest_response = json.loads(response.content)["content"]

                latest_response_first_text = ""
                for member in latest_response:
                    if member["type"] == "text":
                        latest_response_first_text = member["text"]
                print(
                    f"[Model response number {self.turn}] {latest_response_first_text}"
                )
                self.turn += 1

            else:
                raise error


def load_json_file(config_file_path: str):
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)

    return config_data


@click.command()
@click.option(
    "--user-prompt",
    required=True,
    help="The user prompt to give to the agent",
)
@click.option(
    "--conversation-log-file-path",
    default="computer_use_demo/conversation_logs.json",
    help="Where to write the conversation log, in the container's filesystem",
)
@click.option(
    "--max-actions",
    type=int,
    default=100,
    help="The maximum number of loop iterations (each of which is roughly an action by the agent)",
)
@click.option(
    "--model",
    default="arn:aws:bedrock:us-west-2:302263051492:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    help="The model to run that will back the agent. Use 'claude-3-7-sonnet@20250219' for Vertex AI",
)
@click.option(
    "--system-prompt-suffix",
    default="",
    help="A suffix to add to the defaul system prompt",
)
@click.option(
    "--only-n-most-recent-images",
    default=10,
    help="how many images to include at most in a request to the agent",
)
@click.option(
    "--max_tokens",
    default=9048,
    help="Maximum number of tokens",
)
@click.option(
    "--thinking_budget",
    default=4096,
    help="Number of token (out of max_tokens) to allocate for reasoning",
)
def main(
    user_prompt: str,
    conversation_log_file_path: str,
    max_actions: int,
    system_prompt_suffix: str,
    model: str,
    only_n_most_recent_images: int,
    max_tokens: int,
    thinking_budget: int
):
    if "3-7" not in model and "3.7" not in model:
        raise ValueError("This script is only valid for claude sonnet-3.7-cua, exiting...")

    initial_messages = [
        {
            "role": Sender.USER,
            "content": [BetaTextBlockParam(type="text", text=user_prompt)],
        }
    ]

    api_requests_accumulator = APIRequestsLogger(conversation_log_file_path)

    # Get API provider and key
    provider_str = os.getenv("API_PROVIDER", "bedrock")
    provider = APIProvider(provider_str)
    
    # API key is only required for Anthropic provider
    api_key = ""
    if provider == APIProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required when using 'anthropic' provider")
    elif provider == APIProvider.VERTEX:
        # Vertex AI uses Google Cloud authentication (gcloud auth or service account)
        # No API key needed - authentication is handled by google-auth library
        print("Using Google Cloud Vertex AI - ensure you're authenticated with 'gcloud auth application-default login' or have GOOGLE_APPLICATION_CREDENTIALS set")
    elif provider == APIProvider.BEDROCK:
        # Bedrock uses AWS credentials
        print("Using AWS Bedrock - ensure your AWS credentials are configured")

    asyncio.run(
        sampling_loop(
            system_prompt_suffix=system_prompt_suffix,
            model=model,
            provider=provider,
            messages=initial_messages,
            output_callback=_output_callback,
            tool_output_callback=_tool_callback,
            api_response_callback=api_requests_accumulator.log_claude,
            api_key=api_key,
            only_n_most_recent_images=only_n_most_recent_images,
            tool_version="computer_use_20250124",
            max_actions=max_actions,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
    )

try:
    main()
except Exception as e:
    print(f"[!!! Error] An error occurred: {e}. Proceeding to the next task.")
