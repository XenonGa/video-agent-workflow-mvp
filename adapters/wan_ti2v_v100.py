from __future__ import annotations

import argparse
import os
import runpy
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


def cpu_offload_vae_decode(self, z, scale):
    from wan.modules.vae2_2 import unpatchify

    self.clear_cache()
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]

    iter_ = z.shape[2]
    x = self.conv2(z)
    chunks: list[torch.Tensor] = []
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out_chunk = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=True,
            )
        else:
            out_chunk = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
        chunks.append(out_chunk.detach().cpu())
        del out_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = torch.cat(chunks, 2)
    out = unpatchify(out, patch_size=2)
    self.clear_cache()
    return out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version=None,
) -> torch.Tensor:
    del window_size, deterministic, fa_version

    if causal:
        raise NotImplementedError("The V100 SDPA adapter currently supports non-causal Wan attention only.")
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Wan passed sequence lengths to SDPA. The TI2V path normally uses equal, unpadded lengths; "
            "padding masks are not applied by this adapter.",
            stacklevel=2,
        )

    output_dtype = q.dtype
    compute_dtype = dtype
    if q.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        compute_dtype = torch.float16
    if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        compute_dtype = torch.float16

    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2).to(compute_dtype)
    k = k.transpose(1, 2).to(compute_dtype)
    v = v.transpose(1, 2).to(compute_dtype)

    chunk_size = max(1, int(os.environ.get("WAN_SDPA_CHUNK_SIZE", "1024")))
    outputs: list[torch.Tensor] = []
    for start in range(0, q.shape[2], chunk_size):
        q_chunk = q[:, :, start : start + chunk_size]
        output = F.scaled_dot_product_attention(
            q_chunk,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=softmax_scale,
        )
        outputs.append(output)

    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous().to(output_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official Wan2.2 TI2V generator on V100 using chunked PyTorch SDPA."
    )
    parser.add_argument(
        "--wan-root",
        type=Path,
        required=True,
        help="Path to the cloned Wan2.2 source repository.",
    )
    parser.add_argument(
        "--sdpa-chunk-size",
        type=int,
        default=1024,
        help="Query chunk size for SDPA. Lower values reduce peak memory and increase runtime.",
    )
    adapter_args, wan_args = parser.parse_known_args()

    wan_root = adapter_args.wan_root.expanduser().resolve()
    generate_py = wan_root / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"Wan generate.py was not found: {generate_py}")

    os.environ["WAN_SDPA_CHUNK_SIZE"] = str(adapter_args.sdpa_chunk_size)
    sys.path.insert(0, str(wan_root))

    import wan.modules.attention as attention_module
    import wan.modules.model as model_module
    import wan.modules.vae2_2 as vae_module

    attention_module.attention = sdpa_attention
    model_module.attention = sdpa_attention
    vae_module.WanVAE_.decode = cpu_offload_vae_decode

    print(
        f"[wan-v100] FlashAttention bypassed; using chunked PyTorch SDPA "
        f"(chunk={adapter_args.sdpa_chunk_size}).",
        flush=True,
    )
    print("[wan-v100] VAE decode chunks will be offloaded to CPU before concatenation.", flush=True)

    sys.argv = [str(generate_py), *wan_args]
    runpy.run_path(str(generate_py), run_name="__main__")


if __name__ == "__main__":
    main()
