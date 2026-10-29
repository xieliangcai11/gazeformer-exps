"""Precompute 3D gaze directions for 2D-screen datasets with the Gaze360 model.

This is the first stage of Route 2 ("3D prior for 2D calibration"):

    face crop -> Gaze360 3D gaze network -> g in camera coordinates
    bbox center -> approximate face origin o in camera coordinates

The output npz is consumed by ``screen_calib.py``.  It intentionally stores
the input path and frame index so labels can be joined exactly.

Example:
  python -m scripts.calibration.gaze3d_prior \
    --feat-npz out/rgbdgaze/analysis/feats_rgd_test.npz \
    --index-csv data/RGBDGaze/index/index.csv \
    --ckpt out/gaze360/train/checkpoints/best—separate-added_Gaze360.pt \
    --out out/rgbdgaze/analysis/gaze3d_rgd_test.npz
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

import clip

sys.path[:0] = ["src", "."]
from gazelab.models.transformers import TransformerDeepSeek_gaze  # noqa: E402

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_TX = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])
IMAGENET_TX = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def log(msg):
    print(msg, flush=True)


class Gaze360Net(nn.Module):
    """Inference-only copy of the archived Gaze360 CLIP+ResNet model."""

    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.logit_scale = self.model.logit_scale
        base = models.resnet50(weights=None)
        self.main_model = create_feature_extractor(
            base, return_nodes={"layer4": "features"}
        )
        # Present only so the archived checkpoint loads strictly; the 3D
        # prediction comes from the transformer head.
        self.fuse_model = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, 3)
        )
        self.transformer = TransformerDeepSeek_gaze(
            12, 512, 2048, 8, 4, 2, 2, 1024, d_model=768, out_dim=3,
        )
        texts = [
            ["a face with bright light", "a face with low light", "a face with shadows"],
            ["a frontal face", "a profile face"],
            ["a face on bright background", "a face on dark background"],
            [
                "A photo of a face looking left",
                "A photo of a face looking upper left",
                "A photo of a face looking up",
                "A photo of a face looking upper right",
                "A photo of a face looking right",
                "A photo of a face looking lower right",
                "A photo of a face looking down",
                "A photo of a face looking lower left",
            ],
        ]
        self.text_tokens = [clip.tokenize(x) for x in texts]

    def prepare_text(self, device):
        """Must run after the checkpoint is loaded into ``self.model``."""
        toks = [t.to(device) for t in self.text_tokens]
        with torch.no_grad():
            self.illum_feats = self.model.encode_text(toks[0]).float()
            self.head_feats = self.model.encode_text(toks[1]).float()
            self.bg_feats = self.model.encode_text(toks[2]).float()
            self.label_tokens = toks[3]
        for name in ("illum_feats", "head_feats", "bg_feats"):
            v = getattr(self, name)
            setattr(self, name, v / v.norm(dim=-1, keepdim=True))

    def forward(self, face, other_face):
        hook = {}
        handle = self.model.visual.transformer.register_forward_hook(
            lambda module, inp, out: hook.__setitem__(
                "tokens", inp[0].permute(1, 0, 2)
            )
        )
        try:
            img = self.model.encode_image(face).float()
        finally:
            handle.remove()
        lab = self.model.encode_text(self.label_tokens).float()
        img_n = img / (img.norm(dim=-1, keepdim=True) + 1e-2)
        lab_n = lab / (lab.norm(dim=-1, keepdim=True) + 1e-2)
        ls = self.logit_scale.float().exp()
        si = ls * img_n @ self.illum_feats.T
        sh = ls * img_n @ self.head_feats.T
        sb = ls * img_n @ self.bg_feats.T
        sl = ls * img_n @ lab_n.T
        f1 = img + self.illum_feats[si.argmax(-1)] + \
            self.head_feats[sh.argmax(-1)] + self.bg_feats[sb.argmax(-1)]
        f1 = f1 / (f1.norm(dim=-1, keepdim=True) + 1e-3)
        f2 = img + lab[sl.argmax(-1)]
        f2 = f2 / (f2.norm(dim=-1, keepdim=True) + 1e-3)
        fm = self.main_model(other_face)["features"]
        b, c, h, w = fm.shape
        f3 = fm.view(b, c, h * w).transpose(1, 2)
        tokens = hook["tokens"][:, 1:, :].float()
        raw = {
            "feature_1": f1, "feature_2": f2, "feature_3": f3,
            "token_img_patch": tokens,
        }
        return self.transformer(raw)


def load_checkpoint(model, path):
    z = torch.load(path, map_location="cpu", weights_only=False)
    state = z["model_state_dict"] if "model_state_dict" in z else z
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[3d] checkpoint={path} epoch={z.get('epoch')} "
        f"mean_test_angle={z.get('mean_test_angle')} "
        f"missing={len(missing)} unexpected={len(unexpected)}")
    model_missing = [k for k in missing if not k.startswith("transformer.")]
    if model_missing:
        log(f"[3d] missing keys sample={model_missing[:8]}")
    if unexpected:
        log(f"[3d] unexpected keys sample={unexpected[:8]}")
    if model_missing or unexpected:
        raise RuntimeError("3D checkpoint does not match the inference model")
    tstate = z.get("transformer_state_dict")
    if tstate is None:
        raise RuntimeError("checkpoint has no transformer_state_dict")
    missing, unexpected = model.transformer.load_state_dict(tstate, strict=False)
    log(f"[3d] transformer missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError("3D transformer state does not match")
    model.transformer.to(next(model.model.parameters()).device).eval()
    return z


def load_records(feat_npz, index_csv):
    z = np.load(feat_npz, allow_pickle=True)
    paths = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    need = set(paths)
    meta = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p in need:
                meta[p] = r
    records = []
    for p, fi in zip(paths, fidx):
        r = meta.get(p)
        if r is None:
            records.append((p, int(fi), None))
            continue
        try:
            box = (float(r["bbox_x"]), float(r["bbox_y"]),
                   float(r["bbox_w"]), float(r["bbox_h"]))
        except (KeyError, ValueError):
            records.append((p, int(fi), None))
            continue
        records.append((p, int(fi), box))
    return records


def crop_bbox(img, box, margin):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * margin
    x0, y0 = cx - side / 2.0, cy - side / 2.0
    x1, y1 = cx + side / 2.0, cy + side / 2.0
    x0 = max(0, min(img.width - 1, x0))
    y0 = max(0, min(img.height - 1, y0))
    x1 = max(1, min(img.width, x1))
    y1 = max(1, min(img.height, y1))
    return img.crop((x0, y0, x1, y1))


def origin_from_box(box, img_w, img_h):
    """Approximate face-center camera coordinates.

    The screen pose in the next stage is learnable, so only the relative
    direction is used here.  z=1 fixes the otherwise unknown camera scale.
    """
    x, y, w, h = box
    cx = (x + w / 2.0) / max(img_w, 1)
    cy = (y + h / 2.0) / max(img_h, 1)
    return np.array([cx - 0.5, cy - 0.5, 1.0], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--margin", type=float, default=1.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N frames")
    a = ap.parse_args()

    device = a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    log(f"[3d] start={time.strftime('%F %T')} args={vars(a)} device={device}")
    net = Gaze360Net(device)
    load_checkpoint(net, a.ckpt)
    net.to(device).eval()
    net.prepare_text(device)
    log(f"[3d] model params={sum(p.numel() for p in net.parameters())}")

    records = load_records(a.feat_npz, a.index_csv)
    if a.limit:
        records = records[:a.limit]
    log(f"[3d] records={len(records)} missing_meta={sum(1 for r in records if r[2] is None)}")

    preds, origins, ok = [], [], []
    dt = time.time()
    n_ok = n_bad = 0
    for st in range(0, len(records), a.batch):
        chunk = records[st:st + a.batch]
        faces, others, orgs, good = [], [], [], []
        for p, fi, box in chunk:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                n_bad += 1
                if n_bad <= 5:
                    log(f"[3d] WARNING open failed path={p} err={e}")
                continue
            if box is None:
                n_bad += 1
                if n_bad <= 5:
                    log(f"[3d] WARNING missing bbox path={p}")
                continue
            crop = crop_bbox(im, box, a.margin)
            faces.append(CLIP_TX(crop))
            others.append(IMAGENET_TX(crop))
            orgs.append(origin_from_box(box, im.width, im.height))
            good.append((p, fi))
        if not faces:
            continue
        face = torch.stack(faces).to(device)
        other = torch.stack(others).to(device)
        with torch.no_grad():
            g = net(face, other).float().cpu().numpy()
        if not np.isfinite(g).all():
            raise RuntimeError(f"non-finite 3D gaze batch start={st}")
        preds.append(g)
        origins.append(np.stack(orgs))
        ok.extend(good)
        n_ok += len(good)
        if st % (a.batch * 20) == 0:
            log(f"[3d] batch={st//a.batch} ok={n_ok} bad={n_bad} "
                f"elapsed={time.time()-dt:.1f}s")
    if not preds:
        raise RuntimeError("no valid frames")
    pred = np.concatenate(preds).astype(np.float32)
    origin = np.concatenate(origins).astype(np.float32)
    paths = np.array([p for p, _ in ok], object)
    fidx = np.array([fi for _, fi in ok], np.int64)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pred3d=pred, origin=origin, path=paths, frame_index=fidx)
    log(f"[3d] saved {out} n={len(pred)} bad={n_bad} "
        f"gaze_norm[min/med/max]={np.linalg.norm(pred,axis=1).min():.4f}/"
        f"{np.median(np.linalg.norm(pred,axis=1)):.4f}/"
        f"{np.linalg.norm(pred,axis=1).max():.4f} elapsed={time.time()-dt:.1f}s")


if __name__ == "__main__":
    main()
