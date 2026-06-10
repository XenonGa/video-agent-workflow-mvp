# Video Agent Workflow MVP

This MVP implements the first part of a local video-generation agent workflow:

`prompt -> script JSON -> character sheets with three views -> scene concept images -> storyboard animatics -> shot videos -> audio -> sync -> final video`

It uses:

- LangGraph for orchestration.
- A local OpenAI-compatible LLM endpoint for script and prompt planning.
- ComfyUI HTTP/WebSocket API for image generation.
- Local Python generators for storyboard draft videos and placeholder audio.
- Optional command-template hooks for video, TTS, BGM, SFX, and lip-sync models.

The project is designed for local open-source models. It does not require OpenAI or other paid APIs.

## 1. Create Conda Environment

```powershell
conda env create -f environment.yml
conda activate video-agent-workflow
```

If you prefer pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## 2. Start Local LLM

Use one of these options.

### Option A: Ollama

```powershell
ollama pull qwen3:8b
ollama serve
```

Set `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3:8b
LLM_API_KEY=ollama
```

### Option B: vLLM

Run your Qwen model with an OpenAI-compatible server, then set:

```env
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=Qwen/Qwen3-8B
LLM_API_KEY=local
```

### Option C: LM Studio

Start the local server in LM Studio and set:

```env
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=your-loaded-model
LLM_API_KEY=lm-studio
```

## 3. Start ComfyUI

Start ComfyUI with API enabled:

```powershell
python main.py --listen 127.0.0.1 --port 8188
```

Put at least one image checkpoint in ComfyUI, for example SDXL, FLUX, or Qwen-Image via the matching ComfyUI nodes.

This repo includes `workflows/simple_sdxl_t2i_api.json`, a basic text-to-image API workflow. Edit `.env` and set `COMFY_CHECKPOINT` to your checkpoint filename.

For Qwen-Image or FLUX, export an API-format workflow from ComfyUI and point `COMFY_WORKFLOW_PATH` to that file. Then adjust `COMFY_POSITIVE_NODE_ID`, `COMFY_NEGATIVE_NODE_ID`, and optional size/checkpoint node IDs.

## 4. Configure

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your model and ComfyUI setup.

## 5. Run

```powershell
python -m video_agent_workflow.run "一个老教授在雨夜咖啡馆告诉年轻记者城市地下有一个秘密实验室"
```

Outputs are written to:

```text
outputs/<project_id>/
  script.json
  character_prompts.json
  scene_prompts.json
  storyboard.json
  audio_plan.json
  characters/*.png
  scenes/*.png
  animatics/*/*.mp4
  videos/*/*.mp4
  audio/*.wav
  synced/*.mp4
  final/final_video.mp4
  state.json
```

## 6. Replace Placeholder Backends

The default pipeline runs end-to-end without video/audio generation models:

```env
VIDEO_BACKEND=animatic
VOICE_BACKEND=placeholder
BGM_BACKEND=placeholder
SFX_BACKEND=placeholder
SYNC_BACKEND=passthrough
```

To connect external open-source models, switch a backend to `command` and provide a command template.

### Video Model Command

Example shape:

```env
VIDEO_BACKEND=command
VIDEO_COMMAND=python /models/Wan2.2/generate_i2v.py --shot_json {shot_json} --prompt {prompt} --characters {character_refs} --scene {scene_ref} --animatic {animatic} --output {output}
```

The command must create `{output}`.

### Voice / BGM / SFX Commands

```env
VOICE_BACKEND=command
VOICE_COMMAND=python /models/CosyVoice/infer.py --text {text} --speaker {speaker} --output {output}

BGM_BACKEND=command
BGM_COMMAND=python /models/ACE-Step/infer.py --prompt {prompt} --duration {duration} --output {output}

SFX_BACKEND=command
SFX_COMMAND=python /models/audio_sfx/infer.py --prompt {prompt} --duration {duration} --output {output}
```

Each command must create `{output}` as a WAV file.

### Lip Sync Command

```env
SYNC_BACKEND=command
SYNC_COMMAND=python /models/LatentSync/inference.py --video {video} --audio {audio} --output {output}
```

The command must create `{output}`.

## Model Downloads You May Need

You need to download these yourself depending on your hardware:

- Text model: `Qwen3-8B` or `Qwen3-14B` for Ollama/vLLM/LM Studio.
- Image model in ComfyUI:
  - Easy first run: SDXL checkpoint.
  - Better Chinese/prompt adherence: Qwen-Image with compatible ComfyUI nodes.
  - Better cinematic images: FLUX/FLUX Kontext with compatible ComfyUI nodes.

This MVP does not download models automatically because image/video checkpoints are large and depend heavily on your GPU VRAM.
