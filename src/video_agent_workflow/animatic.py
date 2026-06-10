from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .config import Settings
from .schemas import StoryboardShot
from .utils import ensure_dir, save_json


POSITION_X = {
    "left": 0.28,
    "center": 0.50,
    "right": 0.72,
    "foreground": 0.50,
    "background": 0.50,
}


def generate_animatics(storyboard: list[dict], output_root: Path, settings: Settings) -> list[str]:
    animatic_dir = ensure_dir(output_root / "animatics")
    paths: list[str] = []
    for item in storyboard:
        shot = StoryboardShot.model_validate(item)
        shot_dir = ensure_dir(animatic_dir / shot.shot_id)
        save_json(shot_dir / "shot.json", shot.model_dump())
        first_frame = _draw_frame(shot, settings, frame_index=0)
        first_frame.save(shot_dir / "keyframe.png")
        video_path = shot_dir / f"{shot.shot_id}_animatic.mp4"
        _write_video(shot, video_path, settings)
        paths.append(str(video_path))
    return paths


def _write_video(shot: StoryboardShot, path: Path, settings: Settings) -> None:
    frame_count = max(1, int(shot.duration_seconds * settings.animatic_fps))
    with imageio.get_writer(path, fps=settings.animatic_fps, codec="libx264", quality=7) as writer:
        for index in range(frame_count):
            frame = _draw_frame(shot, settings, index)
            writer.append_data(np.asarray(frame))


def _draw_frame(shot: StoryboardShot, settings: Settings, frame_index: int) -> Image.Image:
    width = settings.animatic_width
    height = settings.animatic_height
    bg = _color_from_text(shot.color_palette)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    horizon = int(height * 0.58)
    draw.rectangle((0, horizon, width, height), fill=_darken(bg, 0.72))
    draw.line((0, horizon, width, horizon), fill=(35, 35, 35), width=3)

    _draw_camera_frame(draw, width, height, shot)
    _draw_props(draw, width, height, shot)
    _draw_characters(draw, width, height, shot, frame_index)
    _draw_labels(draw, width, height, shot)
    return image


def _draw_camera_frame(draw: ImageDraw.ImageDraw, width: int, height: int, shot: StoryboardShot) -> None:
    inset = 36
    if "close" in shot.shot_size.lower():
        inset = 120
    elif "wide" in shot.shot_size.lower():
        inset = 18
    draw.rectangle((inset, inset, width - inset, height - inset), outline=(245, 245, 245), width=4)
    if "low" in shot.camera_angle.lower():
        draw.line((0, height * 0.78, width, height * 0.55), fill=(255, 255, 255), width=2)
    elif "high" in shot.camera_angle.lower():
        draw.line((0, height * 0.35, width, height * 0.60), fill=(255, 255, 255), width=2)


def _draw_props(draw: ImageDraw.ImageDraw, width: int, height: int, shot: StoryboardShot) -> None:
    y = int(height * 0.68)
    for index, prop in enumerate(shot.props[:5]):
        x = int(width * (0.15 + index * 0.16))
        draw.rounded_rectangle((x, y, x + 86, y + 46), radius=8, outline=(20, 20, 20), width=3, fill=(220, 220, 210))
        draw.text((x + 6, y + 14), prop[:10], fill=(0, 0, 0))


def _draw_characters(draw: ImageDraw.ImageDraw, width: int, height: int, shot: StoryboardShot, frame_index: int) -> None:
    for index, item in enumerate(shot.blocking):
        pos = item.position.lower()
        x = int(width * POSITION_X.get(pos, 0.32 + index * 0.22))
        if "foreground" in pos:
            scale = 1.35
        elif "background" in pos:
            scale = 0.62
        elif item.scale == "large":
            scale = 1.15
        elif item.scale == "small":
            scale = 0.75
        else:
            scale = 1.0
        bob = int(np.sin(frame_index / 6) * 4)
        body_h = int(height * 0.32 * scale)
        body_w = int(width * 0.075 * scale)
        base_y = int(height * 0.74) + bob
        head_r = int(body_w * 0.38)
        color = _color_from_text(item.character_id)
        draw.ellipse((x - head_r, base_y - body_h - head_r * 2, x + head_r, base_y - body_h), fill=(230, 220, 205), outline=(20, 20, 20), width=3)
        draw.rounded_rectangle((x - body_w, base_y - body_h, x + body_w, base_y), radius=18, fill=color, outline=(20, 20, 20), width=3)
        draw.line((x - body_w, base_y - int(body_h * 0.45), x - int(body_w * 1.8), base_y - int(body_h * 0.08)), fill=(20, 20, 20), width=4)
        draw.line((x + body_w, base_y - int(body_h * 0.45), x + int(body_w * 1.8), base_y - int(body_h * 0.08)), fill=(20, 20, 20), width=4)
        draw.text((x - body_w, base_y + 8), item.character_id, fill=(0, 0, 0))
        draw.text((x - body_w, base_y + 26), item.action[:16], fill=(0, 0, 0))


def _draw_labels(draw: ImageDraw.ImageDraw, width: int, height: int, shot: StoryboardShot) -> None:
    lines = [
        f"{shot.shot_id} | {shot.shot_size} | {shot.camera_angle}",
        f"move: {shot.camera_movement}",
        f"color: {shot.color_palette}",
    ]
    y = 16
    for line in lines:
        draw.rectangle((16, y - 4, min(width - 16, 16 + len(line) * 8), y + 18), fill=(255, 255, 255))
        draw.text((22, y), line, fill=(0, 0, 0))
        y += 26


def _color_from_text(text: str) -> tuple[int, int, int]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return (80 + digest[0] % 100, 80 + digest[1] % 100, 80 + digest[2] % 100)


def _darken(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, int(value * factor)) for value in color)
