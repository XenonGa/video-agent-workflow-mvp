from __future__ import annotations

import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .animatic import generate_animatics
from .audio import generate_audio_assets
from .comfy import ComfyClient
from .config import Settings
from .llm import LocalChatModel
from .prompts import (
    AUDIO_PLAN_SYSTEM,
    AUDIO_PLAN_USER_TEMPLATE,
    CHARACTER_PROMPT_SYSTEM,
    CHARACTER_PROMPT_USER_TEMPLATE,
    SCENE_PROMPT_SYSTEM,
    SCENE_PROMPT_USER_TEMPLATE,
    SCRIPT_SYSTEM,
    SCRIPT_USER_TEMPLATE,
    STORYBOARD_SYSTEM,
    STORYBOARD_USER_TEMPLATE,
)
from .schemas import AudioPlan, ImagePrompt, Script, StoryboardShot, WorkflowState
from .utils import ensure_dir, save_json
from .video import compose_final_video, generate_shot_videos, sync_shot_videos


def build_graph(settings: Settings):
    llm = LocalChatModel(settings)
    comfy = ComfyClient(settings)

    def generate_script(state: WorkflowState) -> WorkflowState:
        print("[1/10] Generating script JSON with local LLM...", flush=True)
        data = llm.json_chat(
            SCRIPT_SYSTEM,
            SCRIPT_USER_TEMPLATE.format(prompt=state["user_prompt"]),
        )
        script = Script.model_validate(data).model_dump()
        project_dir = Path(state["output_dir"])
        save_json(project_dir / "script.json", script)
        print("[1/10] Script saved.", flush=True)
        return {"script": script}

    def generate_character_prompts(state: WorkflowState) -> WorkflowState:
        print("[2/10] Generating character design prompts...", flush=True)
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            CHARACTER_PROMPT_SYSTEM,
            CHARACTER_PROMPT_USER_TEMPLATE.format(script_json=script_json),
        )
        prompts = [ImagePrompt.model_validate(item).model_dump() for item in data]
        save_json(Path(state["output_dir"]) / "character_prompts.json", prompts)
        print("[2/10] Character prompts saved.", flush=True)
        return {"character_prompts": prompts}

    def generate_scene_prompts(state: WorkflowState) -> WorkflowState:
        print("[3/10] Generating scene concept prompts...", flush=True)
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            SCENE_PROMPT_SYSTEM,
            SCENE_PROMPT_USER_TEMPLATE.format(script_json=script_json),
        )
        prompts = [ImagePrompt.model_validate(item).model_dump() for item in data]
        save_json(Path(state["output_dir"]) / "scene_prompts.json", prompts)
        print("[3/10] Scene prompts saved.", flush=True)
        return {"scene_prompts": prompts}

    def generate_character_images(state: WorkflowState) -> WorkflowState:
        print("[4/10] Generating character sheets with ComfyUI...", flush=True)
        output_dir = ensure_dir(Path(state["output_dir"]) / "characters")
        image_paths: list[str] = []
        for item in state["character_prompts"]:
            prompt = ImagePrompt.model_validate(item)
            print(f"[4/10] ComfyUI character image: {prompt.id}", flush=True)
            path = comfy.generate_image(
                prompt.positive_prompt,
                prompt.negative_prompt,
                output_dir,
                f"character_{prompt.id}",
            )
            image_paths.append(str(path))
        print("[4/10] Character images saved.", flush=True)
        return {"character_images": image_paths}

    def generate_scene_images(state: WorkflowState) -> WorkflowState:
        print("[5/10] Generating scene concept images with ComfyUI...", flush=True)
        output_dir = ensure_dir(Path(state["output_dir"]) / "scenes")
        image_paths: list[str] = []
        for item in state["scene_prompts"]:
            prompt = ImagePrompt.model_validate(item)
            print(f"[5/10] ComfyUI scene image: {prompt.id}", flush=True)
            path = comfy.generate_image(
                prompt.positive_prompt,
                prompt.negative_prompt,
                output_dir,
                f"scene_{prompt.id}",
            )
            image_paths.append(str(path))
        print("[5/10] Scene images saved.", flush=True)
        return {"scene_images": image_paths}

    def generate_storyboard_plan(state: WorkflowState) -> WorkflowState:
        print("[6/10] Generating storyboard and blocking plan...", flush=True)
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            STORYBOARD_SYSTEM,
            STORYBOARD_USER_TEMPLATE.format(
                script_json=script_json,
                character_images_json=json.dumps(state["character_images"], ensure_ascii=False, indent=2),
                scene_images_json=json.dumps(state["scene_images"], ensure_ascii=False, indent=2),
            ),
        )
        storyboard = [StoryboardShot.model_validate(item).model_dump() for item in data]
        save_json(Path(state["output_dir"]) / "storyboard.json", storyboard)
        print("[6/10] Storyboard saved.", flush=True)
        return {"storyboard": storyboard}

    def generate_animatic_videos(state: WorkflowState) -> WorkflowState:
        print("[7/10] Generating storyboard draft animatic videos...", flush=True)
        paths = generate_animatics(state["storyboard"], Path(state["output_dir"]), settings)
        print("[7/10] Animatics saved.", flush=True)
        return {"animatic_videos": paths}

    def generate_visual_videos(state: WorkflowState) -> WorkflowState:
        print("[8/10] Generating shot videos...", flush=True)
        paths = generate_shot_videos(
            state["storyboard"],
            state["animatic_videos"],
            state["character_images"],
            state["scene_images"],
            Path(state["output_dir"]),
            settings,
        )
        print("[8/10] Shot videos saved.", flush=True)
        return {"shot_videos": paths}

    def generate_audio_plan(state: WorkflowState) -> WorkflowState:
        print("[9/10] Generating audio design plan...", flush=True)
        script_json = json.dumps(state["script"], ensure_ascii=False, indent=2)
        storyboard_json = json.dumps(state["storyboard"], ensure_ascii=False, indent=2)
        data = llm.json_chat(
            AUDIO_PLAN_SYSTEM,
            AUDIO_PLAN_USER_TEMPLATE.format(script_json=script_json, storyboard_json=storyboard_json),
        )
        plan = AudioPlan.model_validate(data).model_dump()
        save_json(Path(state["output_dir"]) / "audio_plan.json", plan)
        print("[9/10] Audio plan saved.", flush=True)
        return {"audio_plan": plan}

    def generate_audio_and_compose(state: WorkflowState) -> WorkflowState:
        print("[10/10] Generating audio, syncing, and composing final video...", flush=True)
        output_root = Path(state["output_dir"])
        voice_tracks, bgm_track, sfx_tracks, mixed_audio = generate_audio_assets(
            state["audio_plan"], state["storyboard"], output_root, settings
        )
        synced_videos = sync_shot_videos(state["shot_videos"], mixed_audio, output_root, settings)
        final_video = compose_final_video(synced_videos, mixed_audio, output_root)
        print("[10/10] Final video saved.", flush=True)
        return {
            "voice_tracks": voice_tracks,
            "bgm_track": bgm_track,
            "sfx_tracks": sfx_tracks,
            "mixed_audio": mixed_audio,
            "synced_videos": synced_videos,
            "final_video": final_video,
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("generate_script", generate_script)
    graph.add_node("generate_character_prompts", generate_character_prompts)
    graph.add_node("generate_scene_prompts", generate_scene_prompts)
    graph.add_node("generate_character_images", generate_character_images)
    graph.add_node("generate_scene_images", generate_scene_images)
    graph.add_node("generate_storyboard_plan", generate_storyboard_plan)
    graph.add_node("generate_animatic_videos", generate_animatic_videos)
    graph.add_node("generate_visual_videos", generate_visual_videos)
    graph.add_node("generate_audio_plan", generate_audio_plan)
    graph.add_node("generate_audio_and_compose", generate_audio_and_compose)

    graph.add_edge(START, "generate_script")
    graph.add_edge("generate_script", "generate_character_prompts")
    graph.add_edge("generate_character_prompts", "generate_scene_prompts")
    graph.add_edge("generate_scene_prompts", "generate_character_images")
    graph.add_edge("generate_character_images", "generate_scene_images")
    graph.add_edge("generate_scene_images", "generate_storyboard_plan")
    graph.add_edge("generate_storyboard_plan", "generate_animatic_videos")
    graph.add_edge("generate_animatic_videos", "generate_visual_videos")
    graph.add_edge("generate_visual_videos", "generate_audio_plan")
    graph.add_edge("generate_audio_plan", "generate_audio_and_compose")
    graph.add_edge("generate_audio_and_compose", END)
    return graph.compile()
