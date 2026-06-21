# Model Adapters

## Wan2.2 TI2V-5B on NVIDIA V100

`wan_ti2v_v100.py` runs the official Wan2.2 `generate.py` while replacing its
FlashAttention path with chunked PyTorch SDPA. This is intended for Volta GPUs,
where FlashAttention 2 is unsupported.

Example:

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

If SDPA runs out of memory, lower the chunk:

```bash
--sdpa-chunk-size 512
```

This adapter does not install or import FlashAttention. It also converts
attention inputs to FP16 when CUDA BF16 is unavailable. It also offloads VAE
decode chunks to CPU before concatenation, which reduces peak GPU memory at the
end of generation.

## CosyVoice TTS

`cosyvoice_tts.py` provides a small CLI wrapper for CosyVoice / CosyVoice2. It
always writes one WAV file and is intended to be called from `VOICE_COMMAND`.

Zero-shot example:

```bash
/home/gaojx25/envs/cosyvoice/bin/python adapters/cosyvoice_tts.py \
  --model-dir /home/gaojx25/models/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --mode zero-shot \
  --prompt-audio /home/gaojx25/models/CosyVoice/asset/zero_shot_prompt.wav \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --text "你终于来了。城市地下的秘密不能再隐瞒了。" \
  --output /home/gaojx25/test_outputs/cosyvoice_test.wav
```

SFT example, if the model has built-in speakers:

```bash
/home/gaojx25/envs/cosyvoice/bin/python adapters/cosyvoice_tts.py \
  --model-dir /home/gaojx25/models/CosyVoice/pretrained_models/CosyVoice-300M-SFT \
  --model-class cosyvoice \
  --mode sft \
  --speaker 中文女 \
  --text "你终于来了。城市地下的秘密不能再隐瞒了。" \
  --output /home/gaojx25/test_outputs/cosyvoice_sft_test.wav
```

Workflow `.env` example:

```env
VOICE_BACKEND=command
VOICE_COMMAND=/home/gaojx25/envs/cosyvoice/bin/python /home/gaojx25/video_agent_workflow_mvp/adapters/cosyvoice_tts.py --model-dir /home/gaojx25/models/CosyVoice/pretrained_models/CosyVoice2-0.5B --mode zero-shot --prompt-audio /home/gaojx25/models/CosyVoice/asset/zero_shot_prompt.wav --prompt-text "希望你以后能够做的比我还好呦。" --text {text} --output {output}
```
