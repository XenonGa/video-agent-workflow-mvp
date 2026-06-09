from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings
from .utils import extract_json


class LocalChatModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.endpoint = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._tokenizer = None
        self._model = None

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def chat(self, system: str, user: str) -> str:
        if self.settings.llm_provider == "transformers":
            return self._transformers_chat(system, user)
        if self.settings.llm_provider != "openai_compatible":
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.settings.llm_provider}")

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        with httpx.Client(timeout=180) as client:
            response = client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def json_chat(self, system: str, user: str) -> Any:
        return extract_json(self.chat(system, user))

    def _load_transformers(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.settings.llm_model_path:
            raise ValueError("LLM_MODEL_PATH is required when LLM_PROVIDER=transformers")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=transformers requires torch, transformers, and accelerate. "
                "Install them in the conda environment before running."
            ) from exc

        model_path = str(self.settings.llm_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _transformers_chat(self, system: str, user: str) -> str:
        self._load_transformers()
        assert self._model is not None
        assert self._tokenizer is not None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.settings.llm_max_new_tokens,
            temperature=self.settings.llm_temperature,
            do_sample=self.settings.llm_temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
