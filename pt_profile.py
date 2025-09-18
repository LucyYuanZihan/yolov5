# pt_profile.py
import argparse, time, torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression

def calc_gflops(model, imgsz):
    # Try THOP for FLOPs; install with: pip install thop
    try:
        from thop import profile
    except Exception:
        return None
    dev  = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev, dtype=dtype)
    flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
    return flops / 1e9  # GFLOPs per image

@torch.inference_mode()
def benchmark(model, imgsz=640, batch=1, iters=100, warmup=20, nms=False):
    dev  = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    im = torch.zeros(batch, 3, imgsz, imgsz, device=dev, dtype=dtype)

    # warmup
    for _ in range(warmup):
        pred = model.model(im)
        if nms:
            _ = non_max_suppression(pred, 0.25, 0.45, max_det=300)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        pred = model.model(im)
        if nms:
            _ = non_max_suppression(pred, 0.25, 0.45, max_det=300)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    dt = t0 = time.perf_counter() - t0  # elapsed
    fps = iters * batch / dt
    ms  = dt / (iters * batch) * 1000
    return fps, ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=str, nargs="?", default='best.pt')
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--half", action="store_true", help="run FP16 (CUDA only)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--nms", action="store_true", help="include NMS time")
    args = ap.parse_args()

    device = select_device(args.device)
    fp16 = args.half and device.type != "cpu"

    model = DetectMultiBackend(args.weights, device=device, fp16=fp16, fuse=True)
    imgsz = check_img_size(args.img, s=model.stride)
    # keep model & inputs in the same dtype
    (model.model.half() if fp16 else model.model.float())

    gflops = calc_gflops(model, imgsz)
    fps, ms = benchmark(model, imgsz, args.batch, args.iters, args.warmup, nms=args.nms)

    print(f"File: {args.weights}")
    print(f"Device: {device}, dtype: {next(model.model.parameters()).dtype}")
    print(f"Image size: {imgsz}, Batch: {args.batch}")
    if gflops is not None:
        print(f"FLOPs (per image): {gflops:.3f} GFLOPs")
    else:
        print("FLOPs: install 'thop' (pip install thop) to compute GFLOPs.")
    print(f"Speed: {ms:.2f} ms / image  ->  {fps:.2f} FPS  (NMS {'on' if args.nms else 'off'})")
    if gflops is not None:
        print(f"Throughput ≈ {gflops * fps / 1000:.3f} TFLOPS")

if __name__ == "__main__":
    main()
