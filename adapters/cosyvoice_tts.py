from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torchaudio


def _load_model(model_dir: str, model_class: str, fp16: bool):
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

    cls = CosyVoice2 if model_class == "cosyvoice2" else CosyVoice
    kwargs = {
        "load_jit": False,
        "load_trt": False,
        "load_vllm": False,
        "fp16": fp16,
    }
    signature = inspect.signature(cls)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(model_dir, **supported_kwargs)


def _first_result(generator):
    for item in generator:
        return item
    raise RuntimeError("CosyVoice did not return any audio result.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice CLI adapter for the video agent workflow.")
    parser.add_argument("--model-dir", required=True, help="Path to CosyVoice or CosyVoice2 model directory.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--mode", choices=["sft", "zero-shot"], default="zero-shot")
    parser.add_argument("--model-class", choices=["cosyvoice", "cosyvoice2"], default="cosyvoice2")
    parser.add_argument("--speaker", default="", help="Speaker id for SFT mode.")
    parser.add_argument("--prompt-audio", default="", help="Reference prompt WAV for zero-shot mode.")
    parser.add_argument("--prompt-text", default="", help="Transcript of the prompt WAV for zero-shot mode.")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model_dir, args.model_class, args.fp16)

    if args.mode == "sft":
        if not args.speaker:
            raise ValueError("--speaker is required when --mode=sft")
        result = _first_result(model.inference_sft(args.text, args.speaker, stream=False))
    else:
        if not args.prompt_audio or not args.prompt_text:
            raise ValueError("--prompt-audio and --prompt-text are required when --mode=zero-shot")
        from cosyvoice.utils.file_utils import load_wav

        prompt_speech_16k = load_wav(args.prompt_audio, 16000)
        result = _first_result(
            model.inference_zero_shot(
                args.text,
                args.prompt_text,
                prompt_speech_16k,
                stream=False,
            )
        )

    torchaudio.save(str(output_path), result["tts_speech"].cpu(), model.sample_rate)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
