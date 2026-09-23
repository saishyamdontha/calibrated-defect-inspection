"""Run an exported model on images without the web server: print decisions and latency,
and save heatmap overlays.

Usage:
  python -m src.demo_batch --model models/bottle_wrn50.pt --images data/mvtec/bottle/test/good/000.png
  python -m src.demo_batch --model models/screw_dinov2_s_s448.pt --images "data/mvtec/screw/test/good/00*.png" --device cpu --repeat 5
"""
import argparse
import base64
import glob
import statistics
from pathlib import Path

from PIL import Image

from src.inference import Inspector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--repeat", type=int, default=1, help="repeat each image to measure latency")
    p.add_argument("--out-dir", default="results/demo")
    a = p.parse_args()

    ins = Inspector(a.model, device=a.device)
    paths = []
    for pattern in a.images:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latencies = []
    print(f"Model {Path(a.model).stem} on {ins.device} | t_ok={ins.t_ok:.3f} t_defect={ins.t_defect:.3f}\n")

    for path in paths:
        img = Image.open(path)
        for _ in range(a.repeat):
            r = ins.inspect(img)
            latencies.append(r["latency_ms"])
        name = f"{Path(a.model).stem}_{Path(path).parent.name}_{Path(path).stem}.png"
        (out / name).write_bytes(base64.b64decode(r["heatmap_png"]))
        print(f"{r['decision']:<7} score {r['score']:7.3f}  {r['latency_ms']:7.1f} ms  {path}")

    lat = sorted(latencies)
    p95 = lat[int(0.95 * (len(lat) - 1))]
    print(f"\nModel-only latency over {len(lat)} calls on {ins.device}: "
          f"median {statistics.median(lat):.1f} ms, p95 {p95:.1f} ms  (heatmaps saved in {out})")


if __name__ == "__main__":
    main()
