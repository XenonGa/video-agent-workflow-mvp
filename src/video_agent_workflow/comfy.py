from __future__ import annotations

import copy
import json
import random
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx
import websocket

from .config import Settings
from .utils import ensure_dir


class ComfyClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.comfy_base_url.rstrip("/")
        parsed = urllib.parse.urlparse(self.base_url)
        self.ws_url = f"ws://{parsed.netloc}/ws"
        self.client_id = str(uuid.uuid4())
        self.workflow = json.loads(Path(settings.comfy_workflow_path).read_text(encoding="utf-8"))

    def generate_image(self, positive: str, negative: str, output_dir: Path, filename_prefix: str) -> Path:
        prompt = self._build_prompt(positive, negative, filename_prefix)
        prompt_id = self._queue_prompt(prompt)
        self._wait_for_prompt(prompt_id)
        images = self._get_output_images(prompt_id)
        if not images:
            raise RuntimeError(f"ComfyUI finished prompt {prompt_id}, but no images were returned.")
        return self._download_image(images[0], output_dir)

    def _build_prompt(self, positive: str, negative: str, filename_prefix: str) -> dict:
        workflow = copy.deepcopy(self.workflow)
        workflow[self.settings.comfy_positive_node_id]["inputs"]["text"] = positive
        workflow[self.settings.comfy_negative_node_id]["inputs"]["text"] = negative

        if self.settings.comfy_checkpoint_node_id:
            workflow[self.settings.comfy_checkpoint_node_id]["inputs"]["ckpt_name"] = self.settings.comfy_checkpoint

        if self.settings.comfy_empty_latent_node_id:
            latent_inputs = workflow[self.settings.comfy_empty_latent_node_id]["inputs"]
            latent_inputs["width"] = self.settings.comfy_width
            latent_inputs["height"] = self.settings.comfy_height

        for node in workflow.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["steps"] = self.settings.comfy_steps
                node["inputs"]["cfg"] = self.settings.comfy_cfg
                node["inputs"]["seed"] = random.randint(1, 2**31 - 1) if self.settings.comfy_seed < 0 else self.settings.comfy_seed

        if self.settings.comfy_save_node_id:
            workflow[self.settings.comfy_save_node_id]["inputs"]["filename_prefix"] = filename_prefix

        return workflow

    def _queue_prompt(self, prompt: dict) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        with httpx.Client(timeout=60) as client:
            response = client.post(f"{self.base_url}/prompt", json=payload)
            response.raise_for_status()
            return response.json()["prompt_id"]

    def _wait_for_prompt(self, prompt_id: str) -> None:
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}", timeout=10)
        try:
            while True:
                message = ws.recv()
                if not isinstance(message, str):
                    continue
                data = json.loads(message)
                if data.get("type") == "executing":
                    payload = data.get("data", {})
                    if payload.get("node") is None and payload.get("prompt_id") == prompt_id:
                        return
        finally:
            ws.close()

    def _get_output_images(self, prompt_id: str) -> list[dict]:
        time.sleep(0.5)
        with httpx.Client(timeout=60) as client:
            response = client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            history = response.json()[prompt_id]

        images: list[dict] = []
        for output in history.get("outputs", {}).values():
            images.extend(output.get("images", []))
        return images

    def _download_image(self, image: dict, output_dir: Path) -> Path:
        ensure_dir(output_dir)
        params = urllib.parse.urlencode(
            {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        target = output_dir / image["filename"]
        with httpx.Client(timeout=120) as client:
            response = client.get(f"{self.base_url}/view?{params}")
            response.raise_for_status()
            target.write_bytes(response.content)
        return target
