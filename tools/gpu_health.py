"""GPU 只读健康预检：绝不初始化 CUDA，因而绝不会把驱动打崩。

为什么必须"只读"
---------------
本项目实测到的故障模式（2026-10-13 两次）：
  某张边缘卡在**负载**下出错 -> nvidia_uvm 卡死(/dev/nvidia-uvm 报 EIO)
  -> 之后**任何**新进程 cuInit 都返回 999 -> 整机 CUDA 不可用，只能重启。
因此"先逐卡压测再决定用哪几张"这个做法本身是破坏性的：压测会把边缘卡推进
不可恢复态，并连带打崩其余卡。压测适合"确认好卡确实好"，**不适合用来找坏卡**。

本工具只用 `nvidia-smi` 的只读查询（NVML，不走 CUDA），实测在整机已卡死的
状态下依然可用。它输出：
  1) 每张卡的句柄可用性、PCIe 链路宽度/代际（链路降级是掉线的前兆）
  2) remapped_rows / retired_pages（"requires reset" 等不可恢复标记）
  3) 温度、功耗、节流原因、显存占用、以及是否被**别人的进程**占用
  4) 一句可直接复制使用的 CUDA_VISIBLE_DEVICES，以及被排除的卡及原因
退出码：有卡被标记为不可用/不可信 => 非 0（便于写进启动前检查）。

用法：
    python tools/gpu_health.py
    python tools/gpu_health.py --min-free-mem-gb 10   # 只保留空闲显存足够的卡
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys



def sh(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr


def valid_fields() -> set[str]:
    """向 nvidia-smi 询问合法的 --query-gpu 字段名（跨驱动版本稳健）。

    踩过的坑：字段名里只要有一个不被驱动识别，整条查询会返回空
    （导致所有卡被误判为"句柄不可用"）。
    """
    rc, out, _ = sh(["nvidia-smi", "--help-query-gpu"])
    return set(re.findall(r'"([A-Za-z0-9_.\[\]]+)"', out))


def query(fields: str) -> dict[int, dict]:
    """按字段查询每张卡；nvidia-smi 取不到句柄的卡不会出现在结果里。"""
    rc, out, err = sh(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    rows = {}
    if not out.strip():
        print(f"[warn] 查询无输出: {err.strip()[:200]}")
    for line in out.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        rows[int(parts[0])] = dict(zip(fields.split(","), parts))
    return rows


def list_handles() -> tuple[dict[int, str], list[str]]:
    """返回 (index->uuid, 取不到句柄的 PCI 地址列表)。"""
    rc, out, err = sh(["nvidia-smi", "-L"])
    ok, bad = {}, []
    for line in (out + "\n" + err).splitlines():
        m = re.match(r"GPU (\d+): .*\(UUID: ([^)]+)\)", line)
        if m:
            ok[int(m.group(1))] = m.group(2)
        m = re.search(r"Unable to determine the device handle for (?:gpu )?"
                      r"([0-9A-Fa-f:.]+)", line)
        if m:
            bad.append(m.group(1))
    return ok, bad


def compute_apps() -> dict[int, list[str]]:
    """哪些卡上有别人的进程（只读）。"""
    rc, out, _ = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader"])
    rc2, out2, _ = sh(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                       "--format=csv,noheader"])
    used_mem = {}
    for line in out.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 2:
            used_mem[parts[0]] = parts[1]
    apps: dict[str, list[str]] = {}
    for line in out2.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 2 and "N/A" not in line:
            pid, uuid = parts[0], parts[1]
            apps.setdefault(uuid, []).append(f"pid {pid} ({used_mem.get(pid, '?')})")
    return apps


def parse_num(s: str, default=float("nan")) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else default


def norm(v: str) -> str:
    """nvidia-smi 的缺失值会长成 `[N/A]`、`[GPU requires reset]`，统一去括号。"""
    return (v or "").strip().strip("[]").strip()


def link_sysfs(busid: str) -> str:
    """从 sysfs 读 PCIe 链路宽度/速率。

    为什么不用 nvidia-smi 的 pci.link.* 字段：部分驱动版本不提供（本机就不支持），
    而 sysfs 一直可读，且在驱动已卡死时仍能读 —— 它正是"掉线前兆"的观测点
    （2026-10-13 中午 GPU5 出现 Bus Type: PCIe -> PCI 的链路降级，随后掉线）。
    """
    if not busid or busid == "?":
        return "?"
    name = busid if busid.startswith("0000:") else "0000:" + busid.split(":", 1)[1]
    base = f"/sys/bus/pci/devices/{name}"
    try:
        cur = open(base + "/current_link_width").read().strip()
        mx = open(base + "/max_link_width").read().strip()
        spd = open(base + "/current_link_speed").read().strip().replace(" ", "")
        return f"x{cur}/x{mx}@{spd}"
    except OSError:
        return "?"


def stack_state() -> tuple[bool, str]:
    """整机 CUDA 栈是否可用（只做最轻量的探测，不分配显存、不跑 kernel）。

    - /dev/nvidia-uvm 可打开（EIO 即已卡死）；
    - cuInit 返回 0。
    这两项是本项目实测的"整机卡死"判据（存在 999 + uvm EIO 的组合）。
    """
    notes = []
    ok = True
    try:
        import os
        os.close(os.open("/dev/nvidia-uvm", os.O_RDONLY))
        notes.append("uvm=OK")
    except OSError as e:
        notes.append(f"uvm=FAIL({e.errno})")
        ok = False
    try:
        import ctypes
        rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
        notes.append(f"cuInit={rc}")
        ok = ok and rc == 0
    except OSError as e:                            # noqa: BLE001
        notes.append(f"cuInit=ERR({e})")
        ok = False
    return ok, " ".join(notes)


def main():
    ap = argparse.ArgumentParser(description="GPU 只读健康预检（不初始化 CUDA）")
    ap.add_argument("--min-free-mem-gb", type=float, default=0.0,
                    help="要求空闲显存不低于该值（默认不限）")
    ap.add_argument("--max-temp", type=float, default=80.0)
    args = ap.parse_args()

    handles, unhandleable = list_handles()
    n_total = len(handles) + len(unhandleable)
    stack_ok, stack_note = stack_state()

    want = ["index", "name", "pci.bus_id", "pci.link.gen.current",
            "pci.link.width.current", "temperature.gpu", "power.draw",
            "power.limit", "utilization.gpu", "memory.used", "memory.total",
            "clocks_throttle_reasons.active", "remapped_rows.pending",
            "remapped_rows.uncorrectable", "retired_pages.pending",
            "retired_pages.uncorrectable"]
    okf = valid_fields()
    dropped = [f for f in want if f not in okf]
    if dropped:
        print(f"[info] 本驱动不支持这些字段，已跳过: {dropped}")
    fields = ",".join([f for f in want if f in okf])
    try:
        q = query(fields)
    except Exception as e:                       # noqa: BLE001
        print(f"查询失败（nvidia-smi 不可用？）: {e}")
        return 2
    apps = compute_apps()

    print("=" * 104)
    print(f"GPU 只读健康预检（不跑 kernel）  物理卡数={n_total} 可取句柄={len(handles)}")
    print(f"整机 CUDA 栈: {'可用' if stack_ok else '**已卡死（任何卡都用不了，需重启）**'}"
          f"   [{stack_note}]")
    print("=" * 104)
    hdr = (f"  {'卡':>3} {'PCI':<12} {'链路':>8} {'温度':>6} {'功耗':>12} "
           f"{'显存占用':>13} {'状态':<28} 备注")
    print(hdr)
    print("-" * 104)

    bad_idx, bad_reason = [], {}
    for idx in range(n_total):
        row = q.get(idx)
        uuid = handles.get(idx)
        if row is None or uuid is None:
            # 用 nvidia-smi -L 报出的地址标识这张卡（索引则按缺失的那个推断）
            addr = unhandleable[0] if len(unhandleable) == 1 else "?"
            print(f"  {idx:>3} {addr:<12} {'—':>8} {'—':>6} {'—':>12} {'—':>13} "
                  f"{'句柄不可用':<28} 驱动无法访问该卡（严重）")
            bad_idx.append(idx)
            bad_reason[idx] = "句柄不可用"
            continue
        link = link_sysfs(row.get("pci.bus_id", "?"))
        temp = parse_num(row.get("temperature.gpu", "N/A"))
        pw_raw = norm(row.get("power.draw", "N/A"))
        pw_lim_raw = norm(row.get("power.limit", "N/A"))
        pw_used = pw_raw.split()[0] if pw_raw and pw_raw != "N/A" else "?"
        pw_lim = pw_lim_raw.split()[0] if pw_lim_raw and pw_lim_raw != "N/A" else "?"
        mem_used = parse_num(row.get("memory.used", "N/A"))
        mem_tot = parse_num(row.get("memory.total", "N/A"))
        if not (mem_used == mem_used and mem_tot == mem_tot):
            mem_used, mem_tot = 0.0, 0.0
        free_gb = (mem_tot - mem_used) / 1024.0
        occ = apps.get(uuid, [])

        flags = []
        # PCIe 链路：实际宽度低于最大宽度 => 链路降级（掉线前兆）
        m = re.match(r"x(\d+)/x(\d+)@(.+)", link)
        if m:
            cur, mx = int(m.group(1)), int(m.group(2))
            if cur != mx:
                flags.append(f"链路降级({link})")
            elif cur not in (1, 2, 4, 8, 16):
                flags.append(f"链路宽度异常({link})")   # 已见过 63/63 = 未定义态
        for f_ in ("remapped_rows.pending", "remapped_rows.uncorrectable",
                   "retired_pages.pending", "retired_pages.uncorrectable"):
            v = norm(row.get(f_, ""))
            if v and v.lower() not in ("no", "n/a", "0", ""):
                flags.append(f"{f_.split('.')[0]}={v}")
        if temp == temp and temp >= args.max_temp:
            flags.append(f"温度{temp:.0f}C")
        if free_gb < args.min_free_mem_gb:
            flags.append(f"空闲{free_gb:.0f}GB不足")
        if occ:
            flags.append("被占用:" + ",".join(occ))
            note_occ = True
        else:
            note_occ = False

        hard = [f for f in flags
                if "降级" in f or "异常" in f or "rows" in f or "pages" in f
                or "句柄" in f]
        # "被占用"只是信息：不影响该卡自身的健康，但影响你能否使用它
        soft_only = [f for f in flags if f not in hard]
        if hard:
            state = "不可用"
        elif note_occ:
            state = "占用中"
        elif soft_only:
            state = "可疑"
        else:
            state = "正常"
        note = " ".join(flags)
        print(f"  {idx:>3} {row.get('pci.bus_id', '?'):<12} {link:>8} {temp:>6.0f} "
              f"{pw_used + '/' + pw_lim:>12} {mem_used / 1024:>6.1f}/{mem_tot / 1024:.0f}GB "
              f"{state:<28} {note}")
        if state == "不可用":
            bad_idx.append(idx)
            bad_reason[idx] = ";".join(hard)

    good = [i for i in range(n_total) if i not in bad_idx]
    print()
    print("=" * 104)
    if not stack_ok:
        print("⚠ 整机 CUDA 栈已卡死：下面的建议**当前不可用**，必须先重启机器。")
    if good:
        print(f"建议可用: CUDA_VISIBLE_DEVICES={','.join(str(i) for i in good)}"
              f"   （{len(good)}/{n_total} 张，仅指卡本身健康）")
    else:
        print("没有可用卡。")
    for i in bad_idx:
        print(f"  排除 GPU {i}: {bad_reason[i]}")
    print()
    print("提醒：不要用『逐卡压测』来找坏卡 —— 负载会把边缘卡推进不可恢复态并连带打崩")
    print("      整机 CUDA（需重启）。本工具只读 NVML，可在整机已卡死时安全使用。")
    return 0 if good and not bad_idx else 1


if __name__ == "__main__":
    sys.exit(main())
