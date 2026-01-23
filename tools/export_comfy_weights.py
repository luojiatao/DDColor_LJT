"""Export a ComfyUI-friendly DDColorMask weight file.

Why this exists
- basicsr training checkpoints are usually saved as a dict wrapper like {"params": state_dict}
- Some ComfyUI custom nodes load weights via: model.load_state_dict(torch.load(path), strict=True)

This script:
- extracts the real state_dict (params/params_ema or direct)
- removes a leading "module." prefix (DDP)
- remaps known key differences:
  - decoder._token_decoder.* -> decoder.color_decoder.*
  - *.token_embed.*          -> *.color_embed.*

Usage
  python tools/export_comfy_weights.py \
    --in  experiments/detail_refine_ddcolormask/models/net_g_30000.pth \
    --out experiments/detail_refine_ddcolormask/models/net_g_30000_comfy.pth
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch


def _pick_state_dict(obj: object) -> dict:
    if isinstance(obj, dict):
        # basicsr checkpoint convention
        for key in ("params", "params_ema", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # might already be a plain state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint format: {type(obj)}")


def export_comfy(in_path: str, out_path: str) -> None:
    ckpt = torch.load(in_path, map_location="cpu")
    sd = _pick_state_dict(ckpt)

    out = OrderedDict()
    for k, v in sd.items():
        key = k
        if key.startswith("module."):
            key = key[7:]

        if key.startswith("decoder._token_decoder."):
            key = key.replace("decoder._token_decoder.", "decoder.color_decoder.", 1)
        key = key.replace(".token_embed.", ".color_embed.")

        out[key] = v.detach().cpu() if hasattr(v, "detach") else v

    torch.save(out, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    export_comfy(args.in_path, args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
