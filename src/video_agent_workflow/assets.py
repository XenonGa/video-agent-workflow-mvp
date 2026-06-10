from __future__ import annotations

from pathlib import Path


def refs_by_id(paths: list[str], prefix: str) -> dict[str, str]:
    refs: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        key = stem
        if stem.startswith(prefix):
            key = stem[len(prefix) :]
        refs[key] = path
    return refs


def scene_ref_for_shot(scene_images: list[str], scene_id: str) -> str:
    refs = refs_by_id(scene_images, "scene_")
    return refs.get(scene_id, scene_images[0] if scene_images else "")


def character_refs_for_shot(character_images: list[str], character_ids: list[str]) -> list[str]:
    refs = refs_by_id(character_images, "character_")
    selected = [refs[item] for item in character_ids if item in refs]
    return selected or character_images
