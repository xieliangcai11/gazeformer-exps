"""导出 RGD base 模型逐帧预测（用于验证集选窗口等离线评测）。

输出列：session,path,frame_no,base_px,base_py,label_x,label_y,em_base_cm
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from scripts.train import build_dataloader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dl = build_dataloader("rgbdgaze", a.split, a.index_dir, a.batch_size, False,
                          orient="raw", eye_crops=True, resolution=a.resolution,
                          num_workers=a.num_workers)
    ds = dl.dataset
    base = ds.base
    paths = [base.rows[int(i)]["rgb_path"] for i in ds.idces]

    m = RGBDGazeDINOv2(use_geom=False, use_eye=True, img_size=a.resolution).to(a.device)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    load_state_dict_any_res(m, st.get("model", st))
    m.eval()

    rows = []
    k = 0
    with torch.no_grad():
        for out, lab in dl:
            face = out.rgb.to(a.device)
            el = out.eye_l.to(a.device)
            er = out.eye_r.to(a.device)
            pred = m(face=face, eye_l=el, eye_r=er).float().cpu().numpy()
            lab = lab.numpy()
            for j in range(len(pred)):
                p = paths[k + j]
                sdir = str(Path(p).parent)
                fno = int(Path(p).stem)
                rows.append((sdir, p, fno, pred[j, 0], pred[j, 1], lab[j, 0], lab[j, 1]))
            k += len(pred)
    # em_base_cm：用索引真实 cm 尺度由外部评测换算，这里先写 0 占位列
    import csv
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "path", "frame_no", "base_px", "base_py",
                    "label_x", "label_y", "em_base_cm"])
        for r in rows:
            w.writerow(list(r) + [""])
    print(f"[dump] {len(rows)} 帧 -> {a.out}")


if __name__ == "__main__":
    main()
