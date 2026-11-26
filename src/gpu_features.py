import math

import cv2
import numpy as np
import torch


def resolve_device(device="auto"):
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Se pidió --device cuda pero PyTorch no ve ninguna GPU. Instala PyTorch con soporte "
            "CUDA (https://pytorch.org/get-started/locally/) y revisa el driver con `nvidia-smi`."
        )
    return dev


def describe_device(dev):
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return f"GPU {p.name} ({p.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda})"
    return "CPU (sin GPU)"


def rgb_to_gray(patches):
    n, h, w, _ = patches.shape
    flat = np.ascontiguousarray(patches).reshape(n * h, w, 3)
    return cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY).reshape(n, h, w)


class GPUFeatureExtractor:

    def __init__(self, lbp_mode="raw", color_bins=8, color_mode="joint",
                 hog_orientations=9, hog_cell=8, device="auto", batch_size=4096):
        if lbp_mode not in ("raw", "hist"):
            raise ValueError(f"lbp_mode inválido: {lbp_mode!r} (usa 'raw' o 'hist')")
        if color_mode not in ("joint", "channels"):
            raise ValueError(f"color_mode inválido: {color_mode!r} (usa 'joint' o 'channels')")
        self.params = dict(lbp_mode=lbp_mode, color_bins=color_bins, color_mode=color_mode,
                           hog_orientations=hog_orientations, hog_cell=hog_cell)
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._cache = {}

    def __call__(self, patch):
        return self.batch(np.asarray(patch)[None])[0]

    def batch(self, patches):
        patches = np.asarray(patches)
        if patches.ndim != 4 or patches.shape[1] != patches.shape[2]:
            raise ValueError("Se esperan parches cuadrados con forma (N, P, P, 3)")
        out = []
        for i in range(0, len(patches), self.batch_size):
            out.append(self._batch_tensor(patches[i:i + self.batch_size]).cpu().numpy())
        if not out:
            return np.zeros((0, self.dim(patches.shape[1])), np.float32)
        return np.concatenate(out)

    def dim(self, patch_size):
        p = self.params
        nc = patch_size // p["hog_cell"]
        nb = max(nc - 1, 0)
        hog_d = nb * nb * 4 * p["hog_orientations"]
        lbp_d = 100 if p["lbp_mode"] == "raw" else 256
        col_d = p["color_bins"] ** 3 if p["color_mode"] == "joint" else 3 * p["color_bins"]
        return hog_d + lbp_d + col_d

    @torch.no_grad()
    def _batch_tensor(self, patches):
        dev = self.device
        gray = torch.from_numpy(rgb_to_gray(patches)).to(dev, non_blocking=True)
        rgb = torch.from_numpy(np.ascontiguousarray(patches)).to(dev, non_blocking=True)
        gray = gray.to(torch.float64)
        feats = [self._hog(gray), self._lbp(gray), self._color(rgb)]
        return torch.cat([f.to(torch.float32) for f in feats], dim=1)

    def _hog(self, img):
        n, s, _ = img.shape
        nb = self.params["hog_orientations"]
        c = self.params["hog_cell"]
        ncell = s // c
        if ncell < 2:
            raise ValueError(f"Parche de {s} px demasiado pequeño para celdas de {c} px y bloques 2x2")

        g_row = torch.zeros_like(img)
        g_col = torch.zeros_like(img)
        g_row[:, 1:-1, :] = img[:, 2:, :] - img[:, :-2, :]
        g_col[:, :, 1:-1] = img[:, :, 2:] - img[:, :, :-2]
        mag = torch.hypot(g_col, g_row)
        ori = torch.rad2deg(torch.atan2(g_row, g_col)) % 180

        w = 180.0 / nb
        edges = torch.tensor([w * i for i in range(1, nb)], dtype=torch.float64, device=img.device)
        bins = (ori.unsqueeze(-1) >= edges).sum(-1)

        m = ncell * c
        mag, bins = mag[:, :m, :m], bins[:, :m, :m]
        cell = torch.arange(m, device=img.device) // c
        cell_idx = cell[:, None] * ncell + cell[None, :]
        flat = (torch.arange(n, device=img.device)[:, None, None] * (ncell * ncell)
                + cell_idx[None]) * nb + bins
        hist = torch.zeros(n * ncell * ncell * nb, dtype=torch.float64, device=img.device)
        hist.scatter_add_(0, flat.reshape(-1), mag.reshape(-1))
        hist = hist.view(n, ncell, ncell, nb) / (c * c)

        blocks = torch.stack([
            torch.stack([hist[:, i:i + ncell - 1, j:j + ncell - 1] for j in range(2)], dim=3)
            for i in range(2)
        ], dim=3)
        eps = 1e-5
        norm = torch.sqrt((blocks ** 2).sum(dim=(3, 4, 5), keepdim=True) + eps ** 2)
        out = torch.minimum(blocks / norm, torch.tensor(0.2, dtype=torch.float64, device=img.device))
        out = out / torch.sqrt((out ** 2).sum(dim=(3, 4, 5), keepdim=True) + eps ** 2)
        return out.reshape(n, -1)

    def _lbp_geometry(self, s, dev):
        key = ("lbp", s, dev)
        if key in self._cache:
            return self._cache[key]
        P, R = 8, 1
        ang = 2 * np.pi * np.arange(P, dtype=np.float64) / P
        rp = np.round(-R * np.sin(ang), 5)
        cp = np.round(R * np.cos(ang), 5)
        r = torch.arange(s, dtype=torch.float64, device=dev)[:, None].expand(s, s)
        c = torch.arange(s, dtype=torch.float64, device=dev)[None, :].expand(s, s)
        geo = []
        for i in range(P):
            rr = r + float(rp[i])
            cc = c + float(cp[i])
            minr, minc = torch.floor(rr), torch.floor(cc)
            maxr, maxc = torch.ceil(rr), torch.ceil(cc)
            dr, dc = rr - minr, cc - minc
            geo.append((minr.long(), minc.long(), maxr.long(), maxc.long(), dr, dc))
        self._cache[key] = geo
        return geo

    def _lbp(self, img):
        n, s, _ = img.shape
        pad = torch.nn.functional.pad(img, (1, 1, 1, 1))

        def px(r, c):
            return pad[:, r + 1, c + 1]

        code = torch.zeros_like(img)
        for i, (r0, c0, r1, c1, dr, dc) in enumerate(self._lbp_geometry(s, img.device)):
            top = (1 - dc) * px(r0, c0) + dc * px(r0, c1)
            bottom = (1 - dc) * px(r1, c0) + dc * px(r1, c1)
            val = (1 - dr) * top + dr * bottom
            code = code + ((val - img) >= 0).to(torch.float64) * (2 ** i)

        if self.params["lbp_mode"] == "raw":
            return code.reshape(n, -1)[:, :100]
        idx = code.reshape(n, -1).long() + torch.arange(n, device=img.device)[:, None] * 256
        hist = torch.bincount(idx.reshape(-1), minlength=n * 256).view(n, 256).to(torch.float64)
        return hist / hist.sum(dim=1, keepdim=True).clamp_min(1)

    def _color(self, rgb):
        n = rgb.shape[0]
        b = self.params["color_bins"]
        lut = torch.tensor([math.floor(v * b / 256) for v in range(256)], device=rgb.device)
        q = lut[rgb.long()].reshape(n, -1, 3)
        if self.params["color_mode"] == "joint":
            idx = q[..., 0] * b * b + q[..., 1] * b + q[..., 2]
            k = b ** 3
            idx = idx + torch.arange(n, device=rgb.device)[:, None] * k
            return torch.bincount(idx.reshape(-1), minlength=n * k).view(n, k).to(torch.float64)
        hists = []
        for ch in range(3):
            idx = q[..., ch] + torch.arange(n, device=rgb.device)[:, None] * b
            hists.append(torch.bincount(idx.reshape(-1), minlength=n * b).view(n, b))
        return torch.cat(hists, dim=1).to(torch.float64)


def make_feature_fn(feature_params, device="auto", batch_size=4096):
    return GPUFeatureExtractor(**feature_params, device=device, batch_size=batch_size)


def batch_features(feature_fn, patches):
    if hasattr(feature_fn, "batch"):
        return feature_fn.batch(patches)
    return np.array([feature_fn(p) for p in patches])
