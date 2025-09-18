# count_pt_info.py  (updated: auto precision + FPS)
import argparse, time
from pathlib import Path
import torch

try:
    from utils.general import non_max_suppression
except Exception:
    non_max_suppression = None

def load_model(weights):
    ckpt = torch.load(weights, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if model is None:
        raise SystemExit("No model object found. If this is a state_dict, rebuild the model from YAML and load_state_dict.")
    if hasattr(model, "fuse"):
        try: model = model.fuse()
        except Exception: pass
    model.eval()
    return model

def pick_precision(model, device, user_prec):
    """Return (model, tensor_dtype) after any casting needed."""
    dev_type = device.type
    # Determine model's current dtype (first floating param)
    mdtype = None
    for p in model.parameters():
        if p.is_floating_point():
            mdtype = p.dtype; break
    if mdtype is None:
        mdtype = torch.float32

    # Normalize user choice
    if user_prec == "auto":
        # On CPU, fp16 not supported -> upcast to fp32 if needed
        if dev_type == "cpu" and mdtype == torch.float16:
            model = model.float(); mdtype = torch.float32
        # On CUDA, keep model dtype as-is
        return model, mdtype

    if user_prec == "float32":
        model = model.float()
        return model, torch.float32

    if user_prec == "float16":
        if dev_type == "cpu":
            # CPU can't run fp16; fall back gracefully
            model = model.float()
            return model, torch.float32
        else:
            model = model.half()
            return model, torch.float16

    # Fallback
    return model, mdtype

@torch.inference_mode()
def benchmark(model, imgsz=640, batch=1, device="0", runs=100, warmup=10, precision="auto", do_nms=False):
    # device
    dev = torch.device("cpu" if device in ("", "cpu") else f"cuda:{int(device)}")
    model.to(dev)
    model, dtype = pick_precision(model, dev, precision)

    # dummy input in the same dtype
    im = torch.zeros(batch, 3, imgsz, imgsz, dtype=dtype, device=dev)

    # warmup
    nw = max(1, warmup)
    for _ in range(nw):
        y = model(im)
        if do_nms and non_max_suppression is not None and isinstance(y, torch.Tensor):
            _ = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45)

    # measure
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        y = model(im)
        if do_nms and non_max_suppression is not None and isinstance(y, torch.Tensor):
            _ = non_max_suppression(y, conf_thres=0.25, iou_thres=0.45)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_images = runs * batch
    total_time = t1 - t0
    fps = total_images / total_time if total_time > 0 else float("inf")
    ms_per_image = (total_time / total_images) * 1000.0
    return fps, ms_per_image, dtype

@torch.inference_mode()
def compute_gflops(model, imgsz=640, device="0", dry_run=True):
    """
    Returns GFLOPs for a single forward pass at the given imgsz (batch=1).
    Uses THOP (MACs * 2 = FLOPs), matches YOLOv5's training summary.
    """
    if thop_profile is None:
        return None  # thop not installed

    dev = torch.device("cpu" if device in ("", "cpu") else f"cuda:{int(device)}")
    m = copy.deepcopy(model).to(dev).eval().float()  # profile in FP32
    im = torch.zeros(1, 3, imgsz, imgsz, device=dev, dtype=torch.float32)

    if dry_run:
        _ = m(im)  # build shapes

    macs, _ = thop_profile(m, inputs=(im,), verbose=False)  # MACs
    gflops = (macs * 2) / 1e9  # MACs->FLOPs
    del m
    return gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default="best.pt")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="0", help='"0" or "cpu"')
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", choices=["auto", "float32", "float16"], default="auto",
                    help="auto=match model dtype (upcast on CPU); float32/float16 to force")
    ap.add_argument("--nms", action="store_true", help="include NMS time")
    args = ap.parse_args()

    w = Path(args.weights)
    model = load_model(w)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_layers = sum(1 for _ in model.modules())

    print(f"File: {w}")
    print(f"Layers: {n_layers}")
    print(f"Parameters (total): {total_params:,}")
    print(f"Parameters (trainable): {trainable_params:,}")
    fps, ms, dtype = benchmark(model, imgsz=args.img, batch=args.batch, device=args.device,
                               runs=args.runs, warmup=args.warmup, precision=args.precision, do_nms=False)
    print(f"\nSpeed (model only): {fps:.2f} FPS  |  {ms:.2f} ms/img  "
          f"[img={args.img}, batch={args.batch}, device={args.device}, dtype={str(dtype).split('.')[-1]}]")

    if args.nms:
        if non_max_suppression is None:
            print("(NMS not available from utils.general, skipping end-to-end timing)")
        else:
            fps2, ms2, _ = benchmark(model, imgsz=args.img, batch=args.batch, device=args.device,
                                     runs=args.runs, warmup=args.warmup, precision=args.precision, do_nms=True)
            print(f"Speed (with NMS):  {fps2:.2f} FPS  |  {ms2:.2f} ms/img")

    if hasattr(model, "info"):
        try: model.info(verbose=False)
        except Exception: pass

if __name__ == "__main__":
    main()
