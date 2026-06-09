# ModelScope Local Model Setup

This guide explains how to use models downloaded from ModelScope with this MVP.

## Qwen3 4B

The current MVP expects a local OpenAI-compatible chat endpoint:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3:4b
LLM_API_KEY=ollama
```

If your Qwen3 model was downloaded from ModelScope as a normal Hugging Face / Transformers folder, you need to serve it first.

### Recommended on Windows: LM Studio

1. Open LM Studio.
2. Import or load the local Qwen3 4B model if LM Studio supports its format.
3. Start the local server.
4. Update `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=<model-name-shown-in-lm-studio>
LLM_API_KEY=lm-studio
```

### Ollama

Ollama cannot directly use a normal ModelScope Transformers folder. It normally needs a GGUF model.

If your downloaded Qwen3 is GGUF, create a `Modelfile`:

```text
FROM E:\path\to\qwen3-4b.gguf
```

Then run:

```powershell
ollama create qwen3-modelscope-4b -f Modelfile
ollama serve
```

Update `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3-modelscope-4b
LLM_API_KEY=ollama
```

### vLLM

vLLM is excellent on Linux/WSL2, but it is usually not the easiest path on native Windows.

Example:

```powershell
python -m vllm.entrypoints.openai.api_server --model E:\path\to\Qwen3-4B --served-model-name qwen3-4b --host 127.0.0.1 --port 8000
```

Update `.env`:

```env
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=qwen3-4b
LLM_API_KEY=local
```

## SDXL

ComfyUI needs an SDXL model it can load.

### If ModelScope gave you a `.safetensors` checkpoint

Copy or symlink it into:

```text
ComfyUI\models\checkpoints\
```

Then update `.env`:

```env
COMFY_CHECKPOINT=<exact-filename>.safetensors
```

Example:

```env
COMFY_CHECKPOINT=sd_xl_base_1.0.safetensors
```

### If ModelScope gave you a Diffusers folder

The folder may look like this:

```text
sdxl/
  model_index.json
  unet/
  vae/
  text_encoder/
  text_encoder_2/
  tokenizer/
  tokenizer_2/
```

The included `workflows/simple_sdxl_t2i_api.json` uses a normal ComfyUI checkpoint loader, so a Diffusers folder will not work with that workflow directly.

You have three options:

1. Download a `.safetensors` SDXL checkpoint and put it in `ComfyUI\models\checkpoints`.
2. Use or create a ComfyUI workflow that loads Diffusers-format SDXL.
3. Convert the Diffusers model to a checkpoint, then put the checkpoint in `ComfyUI\models\checkpoints`.

For the first MVP, option 1 is the least painful.

## Start Services

Start the LLM service, then start ComfyUI:

```powershell
cd <your-comfyui-folder>
python main.py --listen 127.0.0.1 --port 8188
```

Then run:

```powershell
cd E:\video_agent_workflow_mvp
conda activate video-agent-workflow
python -m video_agent_workflow.run "一个老教授在雨夜咖啡馆告诉年轻记者城市地下有一个秘密实验室"
```
