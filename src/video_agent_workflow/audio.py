from __future__ import annotations

import math
import random
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .commands import require_file, run_template_command
from .config import Settings
from .schemas import AudioCue, AudioPlan, DialogueCue, StoryboardShot
from .utils import ensure_dir, save_json


SAMPLE_RATE = 24000


def generate_audio_assets(audio_plan: dict, storyboard: list[dict], output_root: Path, settings: Settings) -> tuple[list[str], str, list[str], str]:
    audio_dir = ensure_dir(output_root / "audio")
    plan = AudioPlan.model_validate(audio_plan)
    save_json(audio_dir / "audio_plan.json", plan.model_dump())

    voice_paths = [_generate_voice(cue, audio_dir, settings) for cue in plan.dialogues]
    total_duration = sum(StoryboardShot.model_validate(item).duration_seconds for item in storyboard)
    bgm_path = _generate_bgm(plan.bgm_prompt, total_duration, audio_dir, settings)
    sfx_paths = [_generate_sfx(cue, audio_dir, settings) for cue in plan.sfx]
    mixed_path = mix_audio_tracks(voice_paths, bgm_path, sfx_paths, audio_dir / "mixed_audio.wav")
    return voice_paths, bgm_path, sfx_paths, mixed_path


def _generate_voice(cue: DialogueCue, audio_dir: Path, settings: Settings) -> str:
    path = audio_dir / "voices" / f"{cue.shot_id}_{_safe(cue.speaker)}.wav"
    ensure_dir(path.parent)
    if settings.voice_backend == "command" and settings.voice_command:
        shot_json = audio_dir.parent / "animatics" / cue.shot_id / "shot.json"
        run_template_command(
            settings.voice_command,
            {"text": cue.text, "speaker": cue.speaker, "output": path, "shot_json": shot_json},
        )
        return str(require_file(path, "voice track"))
    _write_tone(path, cue.duration_seconds, frequency=180 + len(cue.text) % 160, amplitude=0.12)
    return str(path)


def _generate_bgm(prompt: str, duration: float, audio_dir: Path, settings: Settings) -> str:
    path = audio_dir / "bgm.wav"
    if settings.bgm_backend == "command" and settings.bgm_command:
        run_template_command(settings.bgm_command, {"prompt": prompt, "duration": duration, "output": path, "shot_json": ""})
        return str(require_file(path, "BGM track"))
    _write_tone(path, duration, frequency=82, amplitude=0.035)
    return str(path)


def _generate_sfx(cue: AudioCue, audio_dir: Path, settings: Settings) -> str:
    path = audio_dir / "sfx" / f"{cue.shot_id}_{cue.cue_type}_{abs(hash(cue.prompt)) % 10000}.wav"
    ensure_dir(path.parent)
    if settings.sfx_backend == "command" and settings.sfx_command:
        shot_json = audio_dir.parent / "animatics" / cue.shot_id / "shot.json"
        run_template_command(
            settings.sfx_command,
            {"prompt": cue.prompt, "duration": cue.duration_seconds, "output": path, "shot_json": shot_json},
        )
        return str(require_file(path, "SFX track"))
    _write_noise(path, cue.duration_seconds, amplitude=0.03)
    return str(path)


def mix_audio_tracks(voice_paths: list[str], bgm_path: str, sfx_paths: list[str], output_path: Path) -> str:
    ensure_dir(output_path.parent)
    tracks = [bgm_path, *voice_paths, *sfx_paths]
    existing = [path for path in tracks if path and Path(path).exists()]
    if not existing:
        _write_tone(output_path, 1.0, frequency=1, amplitude=0.0)
        return str(output_path)

    if shutil.which("ffmpeg") and len(existing) > 1:
        command = ["ffmpeg", "-y"]
        for path in existing:
            command.extend(["-i", path])
        command.extend(["-filter_complex", f"amix=inputs={len(existing)}:duration=longest:normalize=0", str(output_path)])
        subprocess.run(command, check=True)
    elif len(existing) == 1:
        shutil.copyfile(existing[0], output_path)
    else:
        _mix_wavs(existing, output_path)
    return str(output_path)


def _write_tone(path: Path, duration: float, frequency: float, amplitude: float) -> None:
    duration = max(0.2, duration)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave_data = amplitude * np.sin(2 * math.pi * frequency * t)
    _write_wav(path, wave_data)


def _write_noise(path: Path, duration: float, amplitude: float) -> None:
    duration = max(0.2, duration)
    rng = random.Random(str(path))
    data = np.array([rng.uniform(-amplitude, amplitude) for _ in range(int(SAMPLE_RATE * duration))])
    _write_wav(path, data)


def _write_wav(path: Path, data: np.ndarray) -> None:
    ensure_dir(path.parent)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _mix_wavs(paths: list[str], output_path: Path) -> None:
    arrays: list[np.ndarray] = []
    for path in paths:
        with wave.open(path, "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            arrays.append(np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767)
    length = max(len(item) for item in arrays)
    mix = np.zeros(length, dtype=np.float32)
    for item in arrays:
        mix[: len(item)] += item
    mix = mix / max(1.0, np.max(np.abs(mix)))
    _write_wav(output_path, mix)


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:40] or "voice"
