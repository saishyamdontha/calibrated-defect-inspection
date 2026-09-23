"""Simplified PatchCore: patch features from a frozen backbone, a coreset memory bank of
normal patches, and nearest-neighbour distance as the anomaly score.
"""
import math

import timm
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class CNNFeatureExtractor(torch.nn.Module):
    """WideResNet-50 layer2 + layer3 features with 3x3 local neighbourhood averaging."""

    name = "wrn50"

    def __init__(self, model_name="wide_resnet50_2"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(2, 3))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        f2, f3 = self.backbone(x)
        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, C, H, W)


class PatchCore:
    def __init__(self, extractor, device, coreset_ratio=0.1, proj_dim=128, seed=0):
        self.extractor = extractor.to(device).eval()
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.proj_dim = proj_dim
        self.seed = seed
        self.memory_bank = None

    @torch.no_grad()
    def _embed(self, images):
        feats = self.extractor(images.to(self.device, non_blocking=True))
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)

    @torch.no_grad()
    def fit(self, loader):
        chunks = []
        for batch in loader:
            emb, _ = self._embed(batch["image"])
            chunks.append(emb.reshape(-1, emb.shape[-1]))
        bank = torch.cat(chunks)
        self.n_patches_total = bank.shape[0]
        self.memory_bank = self._coreset(bank)
        return self

    @torch.no_grad()
    def _coreset(self, bank):
        """Greedy k-center coreset on a random projection of the features."""
        n = bank.shape[0]
        k = max(1, int(n * self.coreset_ratio))
        if k >= n:
            return bank
        gen = torch.Generator().manual_seed(self.seed)
        proj = (torch.randn(bank.shape[1], self.proj_dim, generator=gen) / math.sqrt(self.proj_dim)).to(bank.device)
        reduced = bank @ proj
        first = int(torch.randint(n, (1,), generator=gen))
        selected = [first]
        min_dist = torch.linalg.norm(reduced - reduced[first], dim=1)
        for _ in range(k - 1):
            idx = int(torch.argmax(min_dist))
            selected.append(idx)
            min_dist = torch.minimum(min_dist, torch.linalg.norm(reduced - reduced[idx], dim=1))
        return bank[torch.tensor(selected, device=bank.device)]

    @torch.no_grad()
    def _nn_dist(self, queries, chunk=4096):
        out = []
        for i in range(0, queries.shape[0], chunk):
            d = torch.cdist(queries[i:i + chunk], self.memory_bank)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, images, image_size):
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")
        emb, (h, w) = self._embed(images)
        b, _, c = emb.shape
        patch_scores = self._nn_dist(emb.reshape(-1, c)).reshape(b, h, w)
        amap = F.interpolate(patch_scores.unsqueeze(1), size=tuple(image_size), mode="bilinear", align_corners=False)
        amap = gaussian_blur(amap, kernel_size=33, sigma=4.0).squeeze(1)
        image_scores = patch_scores.reshape(b, -1).max(dim=1).values
        return image_scores.cpu(), amap.cpu()
