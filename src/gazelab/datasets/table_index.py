"""内存友好的 CSV 索引（列式存储，按需构造行字典）。

为什么需要
----------
各数据集的索引 CSV 之前统一用 `list(csv.DictReader(f))` 读进来，即"每行一个 dict"。
实测开销：

    RGBDGaze 全量 142,669 行 -> 219.5 MB（1,538 B/行）
    GazeCapture 样本 4,665 行 ->   9.0 MB（1,927 B/行）
    外推到 GazeCapture train 划分 1,251,935 行 -> 约 2.41 GB / 进程

训练时 DataLoader 用 fork 派生 `num_workers` 个 worker 进程。CPython 的引用计数
会在读取对象时写入引用计数字段，从而触发**写时复制**把页复制成私有副本 —— 于是
2.41 GB 会随 worker 数量成倍膨胀（8 卡 x 14 worker = 112 个 worker）。这是多卡
训练卡顿的主因，与 GPU 无关。

本类的做法：按列存成编码/定长数组，`rows[i]` 在真正被访问时才构造 dict。
numpy 数组在 fork 后基本不产生 COW（下标访问不会改动父数组），故内存与 COW 双重收益。

**所有列一律以字符串语义保存**（与 csv.DictReader 完全一致），数值转换仍由调用方
用 `float()/int()` 完成。这一点是刻意的：曾尝试把"整列都能转 float"的列存成
float64，结果 `subject="00002"` 变成 `2.0`，前导零丢失、分组键被破坏。索引里
存在零填充 ID 这类"看似数值、实为标识"的列，自动数值化不安全。

对外接口与 `list[dict]` 兼容：`len()`、`rows[i][col]`、`for r in rows` 均可用。
"""

import csv
from pathlib import Path

import numpy as np

#: 字符串列取值数不超过该阈值时改用编码存储（设备名/活动/split 等只有几十个取值）
CODE_MAX_UNIQUE = 256


class TableIndex:
    """列式 CSV 索引，接口与 list[dict] 兼容（取值一律为 str）。

    两遍扫描：第一遍只统计各列唯一值个数，第二遍才落成数组。
    这样峰值内存 ≈ 数组本身，不会出现"先攒几千万个 Python 字符串再转换"的中间态。
    """

    def __init__(self, path, code_max_unique: int = CODE_MAX_UNIQUE):
        path = Path(path)
        self.path = path

        # ---- 第一遍：只统计 ----
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            self._names = tuple(next(rd))
            ncol = len(self._names)
            n = 0
            uniques = [set() for _ in range(ncol)]
            maxlen = [0] * ncol
            for row in rd:
                if len(row) != ncol:
                    continue                      # 跳过坏行
                n += 1
                for j, v in enumerate(row):
                    if len(uniques[j]) <= code_max_unique:
                        uniques[j].add(v)
                    b = len(v.encode())
                    if b > maxlen[j]:
                        maxlen[j] = b
        self._n = n

        # ---- 第二遍：按列类型落成数组 ----
        plan = []
        for j, name in enumerate(self._names):
            kind = "code" if len(uniques[j]) <= code_max_unique else "bytes"
            plan.append((kind, name, j))

        bufs = [None] * ncol
        luts = [None] * ncol
        for kind, _, j in plan:
            if kind == "code":
                # 唯一值表在第一遍已完整收集（超过阈值的列会被归为 bytes）
                luts[j] = {v: k for k, v in enumerate(sorted(uniques[j]))}
                bufs[j] = np.empty(n, dtype=np.int32)
            else:
                # 预分配定宽 bytes 数组：避免"先攒一个几百万元素的字符串列表
                # 再 np.array()"带来的瞬时峰值（实测占构造峰值的绝大部分）
                bufs[j] = np.empty(n, dtype=f"S{max(maxlen[j], 1)}")

        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            next(rd)
            i = 0
            for row in rd:
                if len(row) != ncol:
                    continue
                for kind, _, j in plan:
                    if kind == "code":
                        bufs[j][i] = luts[j][row[j]]
                    else:
                        bufs[j][i] = row[j].encode()
                i += 1

        self._data = {}
        for kind, name, j in plan:
            if kind == "code":
                self._data[name] = ("code", bufs[j], tuple(sorted(uniques[j])))
            else:
                self._data[name] = ("bytes", bufs[j])

    # ---- 与 list[dict] 兼容的接口 ----
    @property
    def fieldnames(self):
        return self._names

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        out = {}
        for name in self._names:
            d = self._data[name]
            if d[0] == "code":
                out[name] = d[2][d[1][i]]
            else:
                out[name] = d[1][i].decode()
        return out

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    # ---- 额外能力（供训练/评估脚本避免再建大字典） ----
    def lookup(self, name):
        """返回 {列值: 行号}，用于把清单里的路径映射成行下标。

        比 `{r[name]: i for i, r in enumerate(rows)}` 省内存：后者会为每一行
        构造一个临时 dict（125 万行 -> 瞬时约 2.4GB 峰值）。
        """
        d = self._data[name]
        if d[0] == "code":
            uniq = d[2]
            return {uniq[c]: i for i, c in enumerate(d[1])}
        return {v.decode(): i for i, v in enumerate(d[1])}

    def float_column(self, name, dtype=np.float32):
        """把一列按浮点语义整列取出为 numpy 数组。

        等价于 `np.array([float(r[name]) for r in rows], dtype=dtype)`，但**不逐行构造
        dict** —— 后者在 14 万 / 149 万行时是启动期的主要开销（每个 dict 解析一次 CSV 行）。

        列一律以字符串语义保存，故解析仍必需；差别只是省掉了建 dict 与逐行 Python 调用：
        编码列走"唯一值表转 float + 索引数组"完全向量化，bytes 列只做一次列表解析。
        实测量级：RGBDGaze 142,669 行由秒级降到几十毫秒；GazeCapture 149 万行收益更大。
        """
        kind = self._data[name]
        if kind[0] == "code":
            _, codes, uniques = kind
            lut = np.array([float(u) for u in uniques], dtype=dtype)
            return lut[codes]
        _, buf = kind
        return np.array([float(v) for v in buf], dtype=dtype)

    def memory_bytes(self):
        """当前实际占用（用于验证内存修复效果）。"""
        tot = 0
        for d in self._data.values():
            if d[0] == "code":
                tot += d[1].nbytes + sum(len(s) + 49 for s in d[2])
            else:
                tot += d[1].nbytes
        return tot
