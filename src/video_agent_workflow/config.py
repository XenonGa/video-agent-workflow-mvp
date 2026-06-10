from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_model_path: Path | None = None
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.7
    llm_max_new_tokens: int = 4096

    comfy_base_url: str = "http://127.0.0.1:8188"
    comfy_workflow_path: Path = Path("workflows/simple_sdxl_t2i_api.json")
    comfy_positive_node_id: str = "6"
    comfy_negative_node_id: str = "7"
    comfy_checkpoint_node_id: str | None = "4"
    comfy_empty_latent_node_id: str | None = "5"
    comfy_save_node_id: str | None = "9"
    comfy_checkpoint: str = "sd_xl_base_1.0.safetensors"
    comfy_width: int = 1024
    comfy_height: int = 1024
    comfy_steps: int = 30
    comfy_cfg: float = 7.0
    comfy_seed: int = -1

    animatic_width: int = 1280
    animatic_height: int = 720
    animatic_fps: int = 12

    video_backend: str = "animatic"
    video_command: str | None = None
    video_output_ext: str = "mp4"

    voice_backend: str = "placeholder"
    voice_command: str | None = None
    bgm_backend: str = "placeholder"
    bgm_command: str | None = None
    sfx_backend: str = "placeholder"
    sfx_command: str | None = None
    sync_backend: str = "passthrough"
    sync_command: str | None = None
    compose_backend: str = "ffmpeg"

    output_dir: Path = Field(default_factory=lambda: Path("outputs"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
