"""Diagnose "CUDA error: device-side assert triggered" in the Qwen2.5-VL vision tower.

    .venv/bin/python scripts/diagnose_vision.py

Prints the library versions and the visible GPUs, then runs the image processor and the vision
tower on one GPU with two test images, and ends with a VERDICT line. Paste the whole output
when reporting a problem.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # report CUDA errors at the line that caused them

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SIZES = [(640, 480), (300, 500)]


def check_vision(visual, image_processor, device: str, dtype) -> bool:
    """Run the vision tower on two images. True when patches, grid and window index agree
    and the forward pass works."""
    import torch
    from PIL import Image

    enc = image_processor(images=[Image.new("RGB", s, c) for s, c in zip(SIZES, ("red", "blue"))],
                          return_tensors="pt")
    pixels, grid = enc["pixel_values"], enc["image_grid_thw"]
    expected = int(grid.prod(1).sum())
    print(f"  patches: {pixels.shape[0]}  grid_thw: {grid.tolist()}  patches expected from grid: {expected}")
    if pixels.shape[0] != expected:
        print("  PROBLEM: the image processor's patches and grid disagree")
        return False
    visual = visual.to(device=device, dtype=dtype)
    index, _ = visual.get_window_index(grid.to(device))
    limit = pixels.shape[0] // visual.spatial_merge_unit
    print(f"  window index: {index.numel()} entries, max {int(index.max())}, must be below {limit}")
    if index.numel() != limit or int(index.max()) >= limit:
        print("  PROBLEM: the window index does not fit the patches")
        return False
    with torch.no_grad():
        out = visual(pixels.to(device=device, dtype=dtype), grid_thw=grid.to(device))
    print(f"  vision forward OK on {device}: output {tuple(out.shape)}")
    return True


def main() -> int:
    import torch
    import transformers

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__}")
    try:
        import bitsandbytes
        import peft

        print(f"peft {peft.__version__} | bitsandbytes {bitsandbytes.__version__}")
    except Exception as e:  # noqa: BLE001
        print("peft/bitsandbytes import failed:", e)
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')} | visible GPUs: {n}")
    for i in range(n):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    try:
        from indicvlm.train import pin_single_gpu  # noqa: F401

        fix = True
    except Exception:  # noqa: BLE001
        fix = False
    print("single-GPU fix present in indicvlm.train:", fix)
    if n == 0:
        print("VERDICT: no CUDA GPU is visible to PyTorch.")
        return 1

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    print("image processor:", type(processor.image_processor).__name__)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                               device_map={"": "cpu"})
    visual = getattr(model, "visual", None) or model.model.visual
    try:
        ok = check_vision(visual, processor.image_processor, "cuda:0", torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        print(f"  vision forward FAILED on cuda:0: {type(e).__name__}: {str(e).splitlines()[0]}")
        ok = False

    if not ok:
        print("VERDICT: the vision tower fails on a single GPU with these library versions. "
              "This is a library problem, not a multi-GPU one.")
        return 1
    if n > 1 and not fix:
        print(f"VERDICT: the vision tower works on one GPU and {n} GPUs are visible. Training fails because "
              "the Trainer splits each batch across GPUs. Run 'git pull', then train again.")
    elif n > 1:
        print(f"VERDICT: the vision tower works on one GPU, {n} GPUs are visible and the fix is present. "
              "Train again; it should print 'Using GPU 0 only'.")
    else:
        print("VERDICT: the vision tower works on this single GPU, so the crash comes from the training "
              "batches, not from the model or the libraries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
