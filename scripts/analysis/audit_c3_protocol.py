"""C3 协议审计：静态检查 + 指标复算（不训练）。

检查项：
  A 折划分：train/val/test 被试是否互斥；路径数；index/eye_boxes 是否与主索引同一文件
  B 配方：各折 train_args.json 与部署模型（日志）是否一致（分辨率/眼/几何/增强/lr/epochs）
  C 测试集指标复算：EMA 与 oracle 仿射能否复现已知值（1.3770 / 1.1291）
  D 特征健康：NaN/inf、取值范围、是否误含标签字段
  E 逐帧指标复算：CSV 的 em_cm 与 (pred,label,screen_cm) 重算是否一致

用法：
    python -m scripts.analysis.audit_c3_protocol
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from scripts.analysis.context_calib_l2 import load_index, session_features  # noqa: E402

OK, BAD = "PASS", "FAIL"


def main():
    idx = RGBDGaze_INDEX_DIR
    cv = idx / "cv"
    problems = []

    def check(ok, msg):
        print(f"  [{OK if ok else BAD}] {msg}")
        if not ok:
            problems.append(msg)

    # ---------- A 折划分 ----------
    print("=== A. 折划分 ===")
    with open(idx / "index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    off = {}
    for sp in ("train", "val", "test"):
        ps = {x.strip() for x in (idx / f"{sp}.txt").read_text().splitlines() if x.strip()}
        off[sp] = {r["subject"] for r in rows if r["rgb_path"] in ps}
    print(f"  官方划分被试：train {len(off['train'])} / val {len(off['val'])} / test {len(off['test'])}")
    subj_paths = {}
    for r in rows:
        subj_paths.setdefault(r["subject"], set()).add(r["rgb_path"])
    all_paths = {p for s in subj_paths.values() for p in s}
    train_paths = {x.strip() for x in (idx / "train.txt").read_text().splitlines() if x.strip()}

    for k in range(5):
        d = cv / f"fold{k}"
        tr = [x.strip() for x in (d / "train.txt").read_text().splitlines() if x.strip()]
        va = [x.strip() for x in (d / "val.txt").read_text().splitlines() if x.strip()]
        te = [x.strip() for x in (d / "test.txt").read_text().splitlines() if x.strip()]
        check(len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
              and len(set(va) & set(te)) == 0, f"fold{k}: 三份清单路径互斥")
        check(set(tr) | set(va) | set(te) == train_paths,
              f"fold{k}: 并集 = 官方 train 帧集（{len(train_paths)}）")
        # 被试互斥
        def subj_of(paths):
            return {p.split("/decoded/")[0].rsplit("/", 1)[-1] for p in paths}
        st, sv, se = subj_of(tr), subj_of(va), subj_of(te)
        check(not (st & sv) and not (st & se) and not (sv & se),
              f"fold{k}: 被试互斥（train {len(st)} / val {len(sv)} / test {len(se)}）")
        check((st | sv | se) <= off["train"],
              f"fold{k}: 只含官方 train 被试（不碰 val/test）")
        check(not (se & (off["val"] | off["test"])),
              f"fold{k}: test 折不含官方 val/test 被试")
        # index/eye_boxes 是否同一份
        same_idx = (d / "index.csv").resolve() == (idx / "index.csv").resolve() or \
            (d / "index.csv").read_bytes() == (idx / "index.csv").read_bytes()
        check(same_idx, f"fold{k}: index.csv 与主索引一致")
        if (idx / "eye_boxes.npz").is_file():
            same_npz = (d / "eye_boxes.npz").resolve() == (idx / "eye_boxes.npz").resolve() or \
                (d / "eye_boxes.npz").read_bytes() == (idx / "eye_boxes.npz").read_bytes()
            check(same_npz, f"fold{k}: eye_boxes.npz 与主表一致")

    # ---------- B 配方 ----------
    print("\n=== B. CV 配方 vs 部署模型 ===")
    dep_logs = sorted((_ROOT / "out/rgbdgaze/ablation/logs").glob("*eye224_mgpu*"))
    dep = ""
    if dep_logs:
        for line in dep_logs[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(" loss="):
                dep = line.strip()
                break
    print(f"  部署: {dep or '（未找到）'}")
    for k in range(5):
        p = _ROOT / "out/rgbdgaze/ablation/checkpoints" / f"cv_fold{k}" / "train_args.json"
        if not p.is_file():
            check(False, f"fold{k}: 缺 train_args.json")
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        same = (a.get("use_eye") is True and a.get("use_geom") is False
                and float(a.get("geom_randomize") or 0) == 0.0
                and a.get("augment") is True and int(a.get("unfreeze")) == 12
                and a.get("diff_lr") is True and int(a.get("resolution")) == 224)
        same = same and int(a.get("epochs", 0)) == 8 and int(a.get("batch_size", 0)) == 24
        check(same, f"fold{k}: 眼/几何/增强/解冻/分辨率/**epochs=8/batch=24** 与部署一致"
                    f"（eye={a.get('use_eye')} geom={a.get('use_geom')} "
                    f"gr={a.get('geom_randomize')} lr={a.get('lr')} ep={a.get('epochs')}）")

    # ---------- C 测试集指标复算 ----------
    print("\n=== C. 测试集指标复算 ===")
    test_csv = _ROOT / "out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv"
    meta = load_index()
    sess = {}
    with open(test_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sess.setdefault((r["subject"], r["activity"]), []).append(r)
    ema_all, base_all, orc_all, n_all = [], [], [], []
    emes = []
    for k, rs in sess.items():
        rs.sort(key=lambda r: int(float(r["frame_index"])))
        pred = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in rs])
        lab = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rs])
        m = meta.get(rs[0]["path"])
        cm = np.tile(np.array([m[3], m[4]]), (len(rs), 1))
        # E 逐帧 em_cm 复算（抽样）
        for r in rs[:50]:
            e = np.hypot((float(r["pred_x"]) - float(r["label_x"])) * m[3],
                         (float(r["pred_y"]) - float(r["label_y"])) * m[4])
            emes.append(abs(e - float(r["em_cm"])))
        acc = pred[0].copy(); ema = np.empty_like(pred)
        for i in range(len(pred)):
            acc = 0.6 * pred[i] + 0.4 * acc; ema[i] = acc
        em = lambda y: np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                                (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
        s = lab.std(0) / np.maximum(pred.std(0), 1e-6)
        y_o = s * (pred - pred.mean(0)) + lab.mean(0)
        ema_all.append(em(ema) * len(rs)); base_all.append(em(pred) * len(rs))
        orc_all.append(em(y_o) * len(rs)); n_all.append(len(rs))
    fw = lambda xs: float(np.sum(xs) / np.sum(n_all))
    check(abs(fw(ema_all) - 1.3770) < 0.01, f"EMA 帧加权复算 = {fw(ema_all):.4f}（已知 1.3770）")
    check(abs(fw(base_all) - 1.4718) < 0.01, f"基线帧加权复算 = {fw(base_all):.4f}（已知 1.4718）")
    check(abs(fw(orc_all) - 1.1291) < 0.03, f"oracle 仿射复算 = {fw(orc_all):.4f}（已知 1.1291）")
    check(max(emes) < 1e-3, f"逐帧 em_cm 复算最大误差 {max(emes):.2e}")

    # ---------- D 特征健康 ----------
    print("\n=== D. 特征健康 ===")
    F, P = {}, {}
    for k, rs in sess.items():
        F[k], P[k] = session_features(rs, meta)
    keys = sorted(F)
    fk = [x for x in F[keys[0]] if not x.startswith("target")]
    check(all("label" not in x and "target" not in x for x in fk),
          "特征集合不含 label/target 字段")
    X = np.array([[F[k][x] for x in fk] for k in keys], dtype=np.float64)
    check(np.isfinite(X).all(), "特征无 NaN/inf")
    for name in ("s_x", "s_y"):
        vals = np.array([P[k]["s"][0 if name == "s_x" else 1] for k in keys])
        ok = (vals > 0.3).all() and (vals < 3.0).all()
        check(ok, f"{name} 在合理范围 [{vals.min():.2f}, {vals.max():.2f}]")

    print("\n=== 结论 ===")
    if problems:
        print(f"发现 {len(problems)} 个问题 ✗")
        for p in problems:
            print("  -", p)
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
