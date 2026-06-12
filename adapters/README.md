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
attention inputs to FP16 when CUDA BF16 is unavailable.
