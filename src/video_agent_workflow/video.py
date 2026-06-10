from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .assets import character_refs_for_shot, scene_ref_for_shot
from .commands import require_file, run_template_command
from .config import Settings
from .schemas import StoryboardShot
from .utils import ensure_dir, save_json


def generate_shot_videos(
    storyboard: list[dict],
    animatic_videos: list[str],
    character_images: list[str],
    scene_images: list[str],
    output_root: Path,
    settings: Settings,
) -> list[str]:
    videos_dir = ensure_dir(output_root / "videos")
    outputs: list[str] = []
    for item, animatic in zip(storyboard, animatic_videos, strict=False):
        shot = StoryboardShot.model_validate(item)
        shot_dir = ensure_dir(videos_dir / shot.shot_id)
        shot_json = shot_dir / "shot_package.json"
        character_refs = character_refs_for_shot(character_images, [item.character_id for item in shot.blocking])
        scene_ref = scene_ref_for_shot(scene_images, shot.scene_id)
        output = shot_dir / f"{shot.shot_id}_video.{settings.video_output_ext}"
        save_json(
            shot_json,
            {
                "shot": shot.model_dump(),
                "character_refs": character_refs,
                "scene_ref": scene_ref,
                "animatic": animatic,
                "output": str(output),
            },
        )
        if settings.video_backend == "command" and settings.video_command:
            run_template_command(
                settings.video_command,
                {
                    "shot_json": shot_json,
                    "prompt": shot.visual_prompt,
                    "character_refs": character_refs,
                    "scene_ref": scene_ref,
                    "animatic": animatic,
                    "output": output,
                },
            )
        else:
            shutil.copyfile(animatic, output)
        outputs.append(str(require_file(output, "shot video")))
    return outputs


def sync_shot_videos(shot_videos: list[str], mixed_audio: str, output_root: Path, settings: Settings) -> list[str]:
    synced_dir = ensure_dir(output_root / "synced")
    outputs: list[str] = []
    for video in shot_videos:
        src = Path(video)
        output = synced_dir / src.name.replace("_video", "_synced")
        shot_json = src.parent / "shot_package.json"
        if settings.sync_backend == "command" and settings.sync_command:
            run_template_command(
                settings.sync_command,
                {"video": src, "audio": mixed_audio, "output": output, "shot_json": shot_json},
            )
        else:
            shutil.copyfile(src, output)
        outputs.append(str(require_file(output, "synced video")))
    return outputs


def compose_final_video(video_paths: list[str], audio_path: str, output_root: Path) -> str:
    final_dir = ensure_dir(output_root / "final")
    concat_file = final_dir / "videos.txt"
    output = final_dir / "final_video.mp4"
    concat_file.write_text(
        "\n".join(f"file '{Path(path).resolve().as_posix()}'" for path in video_paths),
        encoding="utf-8",
    )
    if shutil.which("ffmpeg"):
        temp = final_dir / "video_only.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(temp),
            ],
            check=True,
        )
        if audio_path and Path(audio_path).exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(temp), "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "copy", "-c:a", "aac", str(output)],
                check=True,
            )
        else:
            shutil.copyfile(temp, output)
    else:
        shutil.copyfile(video_paths[0], output)
    return str(require_file(output, "final video"))
