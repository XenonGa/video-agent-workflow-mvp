from __future__ import annotations

import argparse
import inspect
from pathlib import Path


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


def _load_prompt_wav(path: str, sample_rate: int = 16000):
    import librosa
    import torch

    wav, _ = librosa.load(Path(path).expanduser(), sr=sample_rate, mono=True)
    return torch.from_numpy(wav).float().unsqueeze(0)


def _patch_cosyvoice_load_wav() -> None:
    def load_wav(path, sample_rate):
        return _load_prompt_wav(str(path), sample_rate)

    import cosyvoice.cli.frontend as frontend_module
    import cosyvoice.utils.file_utils as file_utils_module

    frontend_module.load_wav = load_wav
    file_utils_module.load_wav = load_wav


def _save_wav(path: Path, speech, sample_rate: int) -> None:
    import soundfile as sf

    audio = speech.detach().cpu().float().numpy()
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        else:
            audio = audio.T
    sf.write(path, audio, sample_rate)


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
    parser.add_argument(
        "--prompt-audio-input",
        choices=["path", "tensor"],
        default="path",
        help="Pass reference audio to CosyVoice as a file path or preloaded 16 kHz tensor.",
    )
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

        _patch_cosyvoice_load_wav()
        prompt_audio = (
            _load_prompt_wav(args.prompt_audio, 16000)
            if args.prompt_audio_input == "tensor"
            else str(Path(args.prompt_audio).expanduser())
        )
        result = _first_result(
            model.inference_zero_shot(
                args.text,
                args.prompt_text,
                prompt_audio,
                stream=False,
            )
        )

    _save_wav(output_path, result["tts_speech"], model.sample_rate)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
