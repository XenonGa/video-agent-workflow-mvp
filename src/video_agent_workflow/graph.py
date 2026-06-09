from __future__ import annotations

import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .comfy import ComfyClient
from .config import Settings
from .llm import LocalChatModel
from .prompts import (
    CHARACTER_PROMPT_SYSTEM,
    CHARACTER_PROMPT_USER_TEMPLATE,
    SCENE_PROMPT_SYSTEM,
    SCENE_PROMPT_USER_TEMPLATE,
    SCRIPT_SYSTEM,
    SCRIPT_USER_TEMPLATE,
)
from .schemas import ImagePrompt, Script, WorkflowState
from .utils import ensure_dir, save_json


def build_graph(settings: Settings):
    llm = LocalChatModel(settings)
    comfy = ComfyClient(settings)

    def generate_script(state: WorkflowState) -> WorkflowState:
        data = llm.json_chat(
            SCRIPT_SYSTEM,
            SCRIPT_USER_TEMPLATE.format(prompt=state["user_prompt"]),
        )
        script = Script.model_validate(data).model_dump()
        project_dir = Path(state["output_dir"])
        save_json(project_dir / "script.json", script)
        return {"script": script}

    def generate_character_prompts(state: WorkflowState) -> WorkflowState:
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            CHARACTER_PROMPT_SYSTEM,
            CHARACTER_PROMPT_USER_TEMPLATE.format(script_json=script_json),
        )
        prompts = [ImagePrompt.model_validate(item).model_dump() for item in data]
        save_json(Path(state["output_dir"]) / "character_prompts.json", prompts)
        return {"character_prompts": prompts}

    def generate_scene_prompts(state: WorkflowState) -> WorkflowState:
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            SCENE_PROMPT_SYSTEM,
            SCENE_PROMPT_USER_TEMPLATE.format(script_json=script_json),
        )
        prompts = [ImagePrompt.model_validate(item).model_dump() for item in data]
        save_json(Path(state["output_dir"]) / "scene_prompts.json", prompts)
        return {"scene_prompts": prompts}

    def generate_character_images(state: WorkflowState) -> WorkflowState:
        output_dir = ensure_dir(Path(state["output_dir"]) / "characters")
        image_paths: list[str] = []
        for item in state["character_prompts"]:
            prompt = ImagePrompt.model_validate(item)
            path = comfy.generate_image(
                prompt.positive_prompt,
                prompt.negative_prompt,
                output_dir,
                f"character_{prompt.id}",
            )
            image_paths.append(str(path))
        return {"character_images": image_paths}

    def generate_scene_images(state: WorkflowState) -> WorkflowState:
        output_dir = ensure_dir(Path(state["output_dir"]) / "scenes")
        image_paths: list[str] = []
        for item in state["scene_prompts"]:
            prompt = ImagePrompt.model_validate(item)
            path = comfy.generate_image(
                prompt.positive_prompt,
                prompt.negative_prompt,
                output_dir,
                f"scene_{prompt.id}",
            )
            image_paths.append(str(path))
        return {"scene_images": image_paths}

    graph = StateGraph(WorkflowState)
    graph.add_node("generate_script", generate_script)
    graph.add_node("generate_character_prompts", generate_character_prompts)
    graph.add_node("generate_scene_prompts", generate_scene_prompts)
    graph.add_node("generate_character_images", generate_character_images)
    graph.add_node("generate_scene_images", generate_scene_images)

    graph.add_edge(START, "generate_script")
    graph.add_edge("generate_script", "generate_character_prompts")
    graph.add_edge("generate_script", "generate_scene_prompts")
    graph.add_edge("generate_character_prompts", "generate_character_images")
    graph.add_edge("generate_scene_prompts", "generate_scene_images")
    graph.add_edge("generate_character_images", END)
    graph.add_edge("generate_scene_images", END)
    return graph.compile()
