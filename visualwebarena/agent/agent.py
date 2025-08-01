# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import json
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class ConversationRenderer:
    def __init__(self, config_file: str, result_dir: str):
        with open(config_file, "r") as f:
            _config = json.load(f)
            _config_str = ""
            for k, v in _config.items():
                _config_str += f"{k}: {v}\n"
            _config_str = f"<pre>{_config_str}</pre>\n"
            task_id = _config["task_id"]

        self.file = open(
            Path(result_dir) / f"conversation_render_{task_id}.html",
            "a+",
            encoding="utf-8",
        )
        self.file.truncate(0)

        self.raw_json_file = open(
            Path(result_dir) / f"conversation_raw_{task_id}.jsonl",
            "a+",
            encoding="utf-8",
        )
        self.raw_json_file.truncate(0)

        # Initial HTML structure with styles
        self.file.write(
            """
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Chat Log</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .conversation-turn {
            margin: 20px 0;
            border-top: 2px solid #ccc;
        }
        .collapsible {
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 10px;
            text-align: left;
            outline: none;
            border: none;
            width: 100%;
            font-size: 1.1em;
        }
        .content {
            display: none;
            padding: 0 15px;
            margin: 10px 0;
            overflow: hidden;
        }
        .user-message {
            color: #000; /* Set the text color to black (or another desired color) */
            background-color: #ffe6b3; /* Light orange background */
            padding: 10px; /* Add some padding for better readability */
            border-radius: 5px; /* Optional: round the corners for a nicer look */
        }
        .system-message {
            color: #000; /* Set the text color to black (or another desired color) */
            background-color: #d9edf7; /* Light blue background */
            padding: 10px; /* Add some padding for better readability */
            border-radius: 5px; /* Optional: round the corners for a nicer look */
        }
        .model-response {
            color: #000; /* Set the text color to black (or another desired color) */
            background-color: #e6f5e6; /* Light green background */
            padding: 10px; /* Add some padding for better readability */
            border-radius: 5px; /* Optional: round the corners for a nicer look */
        }
    </style>
    <script>
        function toggleContent(element) {
            var content = element.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        }
    </script>
</head>
<body>
"""
        )

    def write_messages(self, messages):
        self.raw_json_file.write(json.dumps(messages) + "\n")
        self.raw_json_file.flush()

        # Write header and collapsible structure
        self.file.write(
            """
<div class='conversation-turn'>
    <button class='collapsible' onclick='toggleContent(this)'><b>Agent Model Request</b> (Click to Expand)</button>
    <div class='content'>
"""
        )

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if content:
                if role == "system":
                    message_class = 'system-message'
                    message_header = 'System Message'
                    
                elif role == "user":
                    message_class = 'user-message'
                    message_header = 'User Message'
                    
                if type(content) == str:
                    text = content.replace("\n", "<br>").replace(
                        "\t", "&nbsp;&nbsp;&nbsp;&nbsp;"
                    )
                    self.file.write(
                        f"<p class='{message_class}'><b>[{message_header}]</b><br> {text}</p>"
                    )
                else:
                    for part in content:
                        if part.get("type") == "text":
                            text = (
                                part.get("text")
                                .replace("\n", "<br>")
                                .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
                            )
                            self.file.write(
                                f"<p class='{message_class}'><b>[{message_header}]</b><br> {text}</p>"
                            )
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url")
                            if image_url:
                                self.file.write(
                                    f"""
    <button class='collapsible' style="background-color: #ffe6b3; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;" onclick='toggleContent(this)'><b>[Image Included in User Message]</b> (Click to Expand)</button>
    <div class='content'>
        <img src='{image_url}' alt='Image Sent to Value Model' style='max-width: 50%; height: auto;'>
    </div>"""
                                )
        self.file.flush()

    def write_model_response(self, model_name, response):
        # Write model response in the collapsible content
        self.file.write(
            f"<p class='model-response'><b>[Model Message by {model_name}]</b><br> {response}</p></div></div>"
        )
        self.file.flush()

    def close(self):
        # Write ending HTML tags and close file
        self.file.write(
            """
</body>
</html>
"""
        )
        self.file.close()


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn=None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        print(f"lm_config model {lm_config.model}")
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if (
            "gemini" in lm_config.model
            or "gpt-4" in lm_config.model
            and "vision" in lm_config.model
            or "gpt-4o" == lm_config.model
            or "gpt-4o-mini" == lm_config.model
        ) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
        conversation_renderer: ConversationRenderer = None,
        prompt_injection: str = None,
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print("WARNING: Input image provided but no image captioner available.")

        if self.multimodal_inputs:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(trajectory, intent, meta_data)
        lm_config = self.lm_config
        n = 0

        if prompt_injection:
            prompt.append({"role": "user", "content": prompt_injection})

        while True:
            conversation_renderer.write_messages(prompt)

            response = call_llm(lm_config, prompt)

            conversation_renderer.write_model_response(lm_config.model, response)

            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            if output_response:
                print(f"Agent: {response}", flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(response)
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
        )
    else:
        raise NotImplementedError(f"agent type {args.agent_type} not implemented")
    return agent
