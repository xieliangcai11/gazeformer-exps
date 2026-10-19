"""归档：2026-10 会话中产生 1.0337 / 0.8492 / 0.8533 的**原始 inline 脚本**（原样保存，未修改）。

来源：历史会话 01a0e884（标题 "Speed up run, sample 50 images"）的 python 工具调用。
注意：脚本内 `pairs` 含每帧 label_x/label_y，`lx = sorted(p[2] for p in pairs)` 即
"该会话全部真值坐标排序后的列表"，映射与仿射都用到了它 ⇒ 见 HANDOFF 的泄漏更正。
仅作证据留档，勿作为方法实现直接使用。
"""

import csv, math, statistics, collections, sys
sys.path.insert(0, "E:/code/gazeformer-exps")
from scripts.analysis.target_select import farthest_sampling
D = "E:/code/gazeformer-exps/out/rgbdgaze/analysis/"
C = 0.5
rows = collections.defaultdict(list)
for r in csv.DictReader(open(D + "preds_eye224_mgpu_test.csv", newline="", encoding="utf-8")):
    rows[r["path"].rsplit("/",1)[0]].append({
        "px": float(r["pred_x"]), "py": float(r["pred_y"]), "lx": float(r["label_x"]), "ly": float(r["label_y"]),
        "ex": float(r["err_x_cm"]), "ey": float(r["ey_err_cm"]) if False else float(r["err_y_cm"]), "fn": int(r["frame_index"])})
for s in rows: rows[s].sort(key=lambda r: r["fn"])
SC = {s: (statistics.median([r["ex"]/(r["px"]-r["lx"]) for r in rs if abs(r["px"]-r["lx"])>0.01]),
          statistics.median([r["ey"]/(r["py"]-r["ly"]) for r in rs if abs(r["py"]-r["ly"])>0.01])) for s, rs in rows.items()}
def em(rs, f, S):
    return sum(math.hypot((f(i,r)[0]-r["lx"])*S[0], (f(i,r)[1]-r["ly"])*S[1]) for i, r in enumerate(rs))/len(rs)
def affine_from(pairs):
    mp = (statistics.mean(p[0] for p in pairs), statistics.mean(p[1] for p in pairs))
    ml = (statistics.mean(p[2] for p in pairs), statistics.mean(p[3] for p in pairs))
    sd = lambda vs: math.sqrt(sum((v-statistics.mean(vs))**2 for v in vs)/len(vs))
    s = (sd([p[2] for p in pairs])/max(sd([p[0] for p in pairs]),1e-6),
         sd([p[3] for p in pairs])/max(sd([p[1] for p in pairs]),1e-6))
    return s, (ml[0]-(s[0]*(mp[0]-C)+C), ml[1]-(s[1]*(mp[1]-C)+C))
def quantile_from(pairs):
    """从 k 个标定点估两个轴的单调映射（分位数阶梯 ✓）；样本少 ⇒ 线性插值 ✓"""
    xs = sorted(pairs, key=lambda p: p[0]); ys = sorted(pairs, key=lambda p: p[1])
    lx = sorted(p[2] for p in pairs); ly = sorted(p[3] for p in pairs)
    def interp(keyvals, tgt):
        n = len(tgt)
        if n < 3: return lambda v: tgt[n//2]
        def f(v):
            # 在 keyvals 中找位置（线性插值 ✓）
            if v <= keyvals[0]: return tgt[0]
            if v >= keyvals[-1]: return tgt[-1]
            for j in range(n-1):
                if keyvals[j] <= v <= keyvals[j+1]:
                    t = (v-keyvals[j])/max(keyvals[j+1]-keyvals[j],1e-9)
                    return tgt[j]*(1-t)+tgt[j+1]*t
            return tgt[-1]
        return f
    fx = interp([p[0] for p in xs], lx); fy = interp([p[1] for p in ys], ly)
    return fx, fy
acc = {k: [] for k in ("base","aff_all","q_all","aff_20","q_20")}
for s, rs in sorted(rows.items()):
    S = SC[s]; n = len(rs)
    acc["base"].append((em(rs, lambda i,r: (r["px"], r["py"]), S), n))
    pa = [(r["px"], r["py"], r["lx"], r["ly"]) for r in rs]
    s_all, b_all = affine_from(pa)
    acc["aff_all"].append((em(rs, lambda i,r: (s_all[0]*(r["px"]-C)+C+b_all[0], s_all[1]*(r["py"]-C)+C+b_all[1]), S), n))
    ox = sorted(range(n), key=lambda i: rs[i]["px"]); oy = sorted(range(n), key=lambda i: rs[i]["py"])
    lx = sorted(r["lx"] for r in rs); ly = sorted(r["ly"] for r in rs)
    mx = {i: lx[k] for k,i in enumerate(ox)}; my = {i: ly[k] for k,i in enumerate(oy)}
    acc["q_all"].append((em(rs, lambda i,r: (mx[i], my[i]), S), n))
    sel = farthest_sampling(rs, 20)
    ps = [(r["px"], r["py"], r["lx"], r["ly"]) for r in sel]
    s20, b20 = affine_from(ps)
    acc["aff_20"].append((em(rs, lambda i,r: (s20[0]*(r["px"]-C)+C+b20[0], s20[1]*(r["py"]-C)+C+b20[1]), S), n))
    fx, fy = quantile_from(ps)
    acc["q_20"].append((em(rs, lambda i,r: (fx(r["px"]), fy(r["py"])), S), n))
N = sum(w for _, w in acc["base"]); B = sum(a*w for a,w in acc["base"])/N
for k, name in (("base","不校正"),("aff_all","仿射·全部目标点"),("q_all","分位数·全部目标点 ✓ 1.0337"),
                ("aff_20","仿射·20 点"),("q_20","分位数·20 点（插值版 ✓ 产品可行）")):
    v = sum(a*w for a,w in acc[k])/N
    print(f"{name:<38}{v:>9.4f}{100*(v/B-1):>9.1f}%")

