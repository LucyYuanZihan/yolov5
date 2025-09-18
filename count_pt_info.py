# count_pt_info.py
import torch, sys
from pathlib import Path

w = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/train/exp/weights/best.pt")
ckpt = torch.load(w, map_location="cpu")

# Prefer EMA if present (that’s what gets saved as best)
model = None
if isinstance(ckpt, dict):
    model = ckpt.get("ema") or ckpt.get("model")
else:  # sometimes .pt saves the model object directly
    model = ckpt

if model is None:
    raise SystemExit(f"Couldn't find model in {w}. If this is a pure state_dict, "
                     f"you'll need to rebuild the model from its YAML and load the state_dict.")

model.eval()

# Counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_layers = sum(1 for _ in model.modules())

print(f"File: {w}")
print(f"Layers: {n_layers}")
print(f"Parameters (total): {total_params:,}")
print(f"Parameters (trainable): {trainable_params:,}")

# If your model object has the YOLOv5 .info() helper, print the summary too
if hasattr(model, "info"):
    try:
        model.info(verbose=False)  # shows layers/params/GFLOPs
    except Exception as e:
        print(f"(model.info() unavailable: {e})")
