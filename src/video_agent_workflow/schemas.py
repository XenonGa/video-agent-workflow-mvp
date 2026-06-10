from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field, field_validator


class Character(BaseModel):
    id: str
    name: str
    age: str = ""
    gender: str = ""
    appearance: str
    costume: str
    personality: str = ""
    role: str = ""

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, value: object) -> str:
        return "" if value is None else str(value)


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

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def coerce_duration(cls, value: object) -> float:
        if value is None or value == "":
            return 5.0
        return float(value)


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


class CharacterBlocking(BaseModel):
    character_id: str
    position: str = "center"
    action: str = "standing"
    scale: str = "medium"


class StoryboardShot(BaseModel):
    shot_id: str
    scene_id: str
    duration_seconds: float = 5.0
    shot_size: str = "medium shot"
    camera_angle: str = "eye level"
    camera_movement: str = "static"
    blocking: list[CharacterBlocking] = Field(default_factory=list)
    props: list[str] = Field(default_factory=list)
    color_palette: str = "cinematic neutral colors"
    visual_prompt: str
    negative_prompt: str = "low quality, blurry, distorted faces, bad anatomy, watermark"

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def coerce_storyboard_duration(cls, value: object) -> float:
        if value is None or value == "":
            return 5.0
        return float(value)


class DialogueCue(BaseModel):
    shot_id: str
    speaker: str
    text: str
    start_seconds: float = 0.0
    duration_seconds: float = 3.0

    @field_validator("start_seconds", "duration_seconds", mode="before")
    @classmethod
    def coerce_dialogue_time(cls, value: object) -> float:
        if value is None or value == "":
            return 0.0
        return float(value)


class AudioCue(BaseModel):
    shot_id: str
    cue_type: str
    prompt: str
    start_seconds: float = 0.0
    duration_seconds: float = 3.0

    @field_validator("start_seconds", "duration_seconds", mode="before")
    @classmethod
    def coerce_audio_time(cls, value: object) -> float:
        if value is None or value == "":
            return 0.0
        return float(value)


class AudioPlan(BaseModel):
    bgm_prompt: str = "subtle cinematic underscore"
    dialogues: list[DialogueCue] = Field(default_factory=list)
    sfx: list[AudioCue] = Field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    project_id: str
    user_prompt: str
    output_dir: str
    script: dict
    character_prompts: list[dict]
    scene_prompts: list[dict]
    character_images: list[str]
    scene_images: list[str]
    storyboard: list[dict]
    animatic_videos: list[str]
    shot_videos: list[str]
    audio_plan: dict
    voice_tracks: list[str]
    bgm_track: str
    sfx_tracks: list[str]
    mixed_audio: str
    synced_videos: list[str]
    final_video: str
    errors: list[str]
