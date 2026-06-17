# AGENTS.md

This file gives AI coding agents and future collaborators the project context needed to work safely in this repository.

## Project Goal

The user is building a local, open-source video-generation agent workflow. The intended creative pipeline is:

```text
user prompt
-> script
-> character design sheets with three views
-> scene concept images
-> storyboard / animatic draft videos
-> final visual shot videos from text + character refs + scene refs + animatics
-> audio branch from script: voice, BGM, SFX
-> audio/video sync
-> final composed video
```

The workflow is designed to run with local models rather than paid hosted APIs.

## Current Architecture

The main package is `video_agent_workflow`.

Important files:

- `src/video_agent_workflow/run.py`: CLI entry point.
- `src/video_agent_workflow/graph.py`: LangGraph orchestration. This is the main workflow graph.
- `src/video_agent_workflow/prompts.py`: LLM prompts for script, character images, scene images, storyboard, and audio planning.
- `src/video_agent_workflow/schemas.py`: Pydantic schemas for script, prompts, storyboard, audio plan, and graph state.
- `src/video_agent_workflow/llm.py`: local LLM wrapper. Supports OpenAI-compatible endpoints and direct Transformers loading.
- `src/video_agent_workflow/comfy.py`: ComfyUI API client for image generation.
- `src/video_agent_workflow/animatic.py`: local storyboard draft video generator.
- `src/video_agent_workflow/video.py`: shot video generation hook, sync hook, and final composition.
- `src/video_agent_workflow/audio.py`: placeholder voice/BGM/SFX generation and audio mixing.
- `src/video_agent_workflow/commands.py`: command-template runner for external model integrations.
- `adapters/wan_ti2v_v100.py`: Wan2.2 TI2V-5B launcher for NVIDIA V100 that bypasses FlashAttention and uses chunked PyTorch SDPA.

Default end-to-end mode uses placeholder backends after image generation:

```env
VIDEO_BACKEND=animatic
VOICE_BACKEND=placeholder
BGM_BACKEND=placeholder
SFX_BACKEND=placeholder
SYNC_BACKEND=passthrough
```

This lets the whole pipeline run before heavy video/audio models are integrated.

## External Models

Models the user is experimenting with:

- Text/script planning: Qwen3 4B or other local Qwen model.
- Text-to-image / reference images: SDXL through ComfyUI.
- Video: Wan2.2 TI2V-5B. On V100, use `adapters/wan_ti2v_v100.py` rather than raw Wan `generate.py`.
- Voice: CosyVoice.
- BGM: ACE-Step.
- SFX: Stable Audio Open.
- Lip sync: LatentSync.

External models are expected to be called through `.env` command templates:

```env
VIDEO_BACKEND=command
VIDEO_COMMAND=python3.12 /path/to/wrapper.py --shot-json {shot_json} --prompt {prompt} --characters {character_refs} --scene {scene_ref} --animatic {animatic} --output {output}

VOICE_BACKEND=command
VOICE_COMMAND=python3.12 /path/to/tts_wrapper.py --text {text} --speaker {speaker} --output {output}

BGM_BACKEND=command
BGM_COMMAND=python3.12 /path/to/bgm_wrapper.py --prompt {prompt} --duration {duration} --output {output}

SFX_BACKEND=command
SFX_COMMAND=python3.12 /path/to/sfx_wrapper.py --prompt {prompt} --duration {duration} --output {output}

SYNC_BACKEND=command
SYNC_COMMAND=python3.12 /path/to/lipsync_wrapper.py --video {video} --audio {audio} --output {output}
```

Each command must create the requested `{output}` file.

## Server / Hardware Context

The lab server uses NVIDIA V100 GPUs. Important constraints:

- V100 is Volta architecture.
- V100 supports FP16 but not native BF16.
- FlashAttention 2 does not support V100.
- Wan2.2 TI2V-5B can be slow on V100 and should be tested with short runs first.
- Use low test settings first: `frame_num=49`, `sample_steps=10`.
- Wan video frame counts should generally satisfy `4n + 1`.

For Wan2.2 TI2V-5B on V100, use:

```bash
python3.12 adapters/wan_ti2v_v100.py \
  --wan-root /home/gaojx25/models/Wan2.2 \
  --sdpa-chunk-size 1024 \
  --task ti2v-5B \
  --size '1280*704' \
  --frame_num 49 \
  --sample_steps 10 \
  --ckpt_dir /home/gaojx25/models/Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --save_file ./wan_test.mp4 \
  --prompt "A cinematic rainy cafe at night, warm amber lighting"
```

If OOM occurs, reduce `--sdpa-chunk-size` to `512` or `256`.

## Configuration Files

- `.env.example` is committed and documents available settings.
- `.env` is local-only and ignored by Git.
- Never put secrets, machine-specific paths, or model checkpoints in committed config.

Common server settings:

```env
LLM_MAX_NEW_TOKENS=4096
LLM_TEMPERATURE=0.3
COMFY_BASE_URL=http://127.0.0.1:8188
COMFY_WIDTH=768
COMFY_HEIGHT=768
```

If LLM JSON output is truncated or invalid, first increase `LLM_MAX_NEW_TOKENS` and lower `LLM_TEMPERATURE`.

## Files To Avoid Committing

Do not commit:

- `.env`
- `outputs/`
- model directories such as `Qwen/`, `MusePublic/`, `Wan2.2-TI2V-5B/`
- model weights: `*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `*.bin`
- generated videos/audio unless explicitly requested

The `.gitignore` already covers the most important cases.

## Development Notes For Future AI Agents

- Prefer small, incremental changes to the graph and schemas.
- Keep output artifacts under `outputs/<project_id>/`.
- Keep external model integrations behind command templates unless the user asks for a tighter integration.
- When adding a new external model, write a small wrapper script that normalizes its CLI to the placeholders used by `.env`.
- Validate Python syntax with `python -m compileall -q src/video_agent_workflow`.
- For media changes, run a small local smoke test that does not require LLM or ComfyUI when possible.
- Do not assume GitHub is reachable from the lab server; the user may need SSH over port 443 or manual file transfer.
- Be careful with Wan2.2 on V100: do not reintroduce FlashAttention as a hard requirement.
