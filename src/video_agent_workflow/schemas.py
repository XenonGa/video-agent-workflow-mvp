from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class Character(BaseModel):
    id: str
    name: str
    age: str = ""
    gender: str = ""
    appearance: str
    costume: str
    personality: str = ""
    role: str = ""


class Scene(BaseModel):
    id: str
    name: str
    location: str
    time_of_day: str = ""
    mood: str = ""
    visual_description: str
    props: list[str] = Field(default_factory=list)
    color_palette: str = ""


class Shot(BaseModel):
    id: str
    scene_id: str
    description: str
    characters: list[str] = Field(default_factory=list)
    shot_size: str = ""
    camera_angle: str = ""
    duration_seconds: float = 5.0
    dialogue: list[str] = Field(default_factory=list)


class Script(BaseModel):
    title: str
    logline: str
    style: str
    characters: list[Character]
    scenes: list[Scene]
    shots: list[Shot]


class ImagePrompt(BaseModel):
    id: str
    name: str
    positive_prompt: str
    negative_prompt: str = "low quality, blurry, deformed, extra limbs, bad anatomy, watermark, text artifacts"


class WorkflowState(TypedDict, total=False):
    project_id: str
    user_prompt: str
    output_dir: str
    script: dict
    character_prompts: list[dict]
    scene_prompts: list[dict]
    character_images: list[str]
    scene_images: list[str]
    errors: list[str]
