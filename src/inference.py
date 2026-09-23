"""Load an exported model and inspect single images."""
import base64
import io
import math
import time

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms as T

from src.conformal import three_way_decisions
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.patchcore import PatchCore, build_extractor


class Inspector:
    def __init__(self, path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.config = ckpt["config"]
        self.t_ok = float(ckpt["t_ok"])
        self.t_defect = float(ckpt["t_defect"])

        size, resize = self.config["image_size"], self.config["resize"]
        extractor = build_extractor(self.config["backbone"], img_size=size)
        self.model = PatchCore(extractor, self.device, seed=self.config["seed"])
        self.model.memory_bank = ckpt["memory_bank"].to(self.device)

        self.tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.view_tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
        ])
        self._warmup(size)

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _warmup(self, size):
        x = torch.zeros(1, 3, size, size)
        self.model.predict(x, (size, size))
        self._sync()

    @torch.no_grad()
    def inspect(self, image):
        image = image.convert("RGB")
        x = self.tf(image).unsqueeze(0)
        self._sync()
        t0 = time.perf_counter()
        score, amap = self.model.predict(x, x.shape[-2:])
        self._sync()
        latency_ms = 1000 * (time.perf_counter() - t0)

        score = float(score[0])
        amap = amap[0].numpy()
        decision = str(three_way_decisions([score], self.t_ok, self.t_defect)[0])
        return {
            "decision": decision,
            "score": score,
            "t_ok": self.t_ok,
            "t_defect": self.t_defect,
            "latency_ms": latency_ms,
            "device": self.device,
            "heatmap_png": self._overlay(image, amap),
        }

    def _overlay(self, image, amap):
        base = np.asarray(self.view_tf(image), dtype=np.float32) / 255.0
        vmax = max(self.t_defect * 1.2, float(amap.max())) if math.isfinite(self.t_defect) else float(amap.max())
        lo = 0.8 * self.t_ok if math.isfinite(self.t_ok) else float(amap.min())
        heat = colormaps["jet"](np.clip((amap - lo) / max(vmax - lo, 1e-8), 0, 1))[..., :3]
        blend = 0.55 * base + 0.45 * heat
        buf = io.BytesIO()
        Image.fromarray((blend * 255).astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

