"""逐张显卡健康检查（每张卡在**独立子进程**里做持续负载压测）。

为什么需要
----------
多卡训练出现「跑到一半某个 rank exitcode 1、其余 rank 被 SIGTERM」时，
最常见的原因之一是**某张卡硬件不稳定**（掉卡 / Xid 错误 / 显存故障 / 供电或散热问题）。
这类故障与代码无关，且往往：
  · 不一定立刻发生（预热后才出现）
  · 不一定可复现（跑几次才失败一次）
  · 用一次小 matmul 的"自检"抓不到

本工具对每张卡单独起一个子进程，做**持续若干秒的矩阵乘 + 显存填充**，
捕获子进程退出码与输出；任何一张卡失败都不影响其余卡的检查。

判读
----
  · 若某张卡稳定 FAIL，而其余全 OK -> 该卡硬件问题，用 CUDA_VISIBLE_DEVICES 排除它
  · 若全部 OK 但训练仍挂 -> 不是卡的问题，转查主机内存 / /dev/shm / 磁盘 / 代码
  · 顺便看 nvidia-smi 的 Xid 计数（见输出末尾提示）

用法
----
    python tools/check_gpus.py                      # 默认每卡 15 秒、占用 6GB
    python tools/check_gpus.py --seconds 30 --mem-gb 10
    python tools/check_gpus.py --only 4             # 只查 4 号卡
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def worker(gpu_idx: int, seconds: int, mem_gb: float) -> int:
    """子进程内：在单张卡上做持续负载。返回 0 = 通过。"""
    import time
    import traceback

    import torch

    try:
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        free_b, total_b = torch.cuda.mem_get_info(dev)
        print(f"    设备: {name}  显存 空闲 {free_b / 2**30:.1f} / 总 {total_b / 2**30:.1f} GB",
              flush=True)
        if free_b / 2**30 < mem_gb + 1.0:
            print(f"    [跳过] 空闲显存不足 {mem_gb}GB", flush=True)
            return 3

        n = int(mem_gb * 2**30 // 4)
        a = torch.empty(n, device="cuda").uniform_(-1, 1)
        b = torch.empty(1024 * 1024, device="cuda").uniform_(-1, 1)
        c = b.view(1024, 1024)
        t0, iters = time.time(), 0
        while time.time() - t0 < seconds:
            # tanh 把结果压回有界区间：直接用 c = c @ c 会因特征值 >1 指数发散、
            # 溢出成 inf，从而把好卡误判为故障（本工具首版即踩此坑）。
            c = torch.tanh(c @ c * 0.5)
            iters += 1
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(c).all() and torch.isfinite(a.sum()))
        used = torch.cuda.max_memory_allocated() / 2**30
        print(f"    {iters} 次 1024x1024 矩阵乘，峰值显存 {used:.1f} GB，"
              f"数值有限: {ok}", flush=True)
        del a, b, c
        return 0 if ok else 2
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        return 1


def main():
    ap = argparse.ArgumentParser(description="逐张显卡健康检查")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=int, default=15, help="每卡压测时长(秒)")
    ap.add_argument("--mem-gb", type=float, default=6.0, help="每卡占用的显存(GB)")
    ap.add_argument("--only", type=int, default=None, help="只检查这一张可见卡的下标")
    args = ap.parse_args()

    if args.worker is not None:            # 子进程模式
        sys.exit(worker(args.worker, args.seconds, args.mem_gb))

    import torch
    if not torch.cuda.is_available():
        print("CUDA 不可用：请检查驱动 / CUDA_VISIBLE_DEVICES")
        sys.exit(1)

    n = torch.cuda.device_count()
    print("=" * 70)
    print(f"逐卡健康检查：{n} 张可见卡，每卡压测 {args.seconds}s / 占用 {args.mem_gb}GB")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置，全部可见)')}")
    print("=" * 70)

    targets = [args.only] if args.only is not None else list(range(n))
    bad = []
    for i in targets:
        print(f"\n[可见卡 {i}] 启动子进程压测…", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        p = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--worker", "0", "--seconds", str(args.seconds),
             "--mem-gb", str(args.mem_gb)],
            env=env, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.stderr.strip():
            print("    stderr:", p.stderr.strip()[:800])
        rc = p.returncode
        print(f"    -> 退出码 {rc} "
              f"{'OK' if rc == 0 else ('跳过(显存不足)' if rc == 3 else '**FAIL**')}")
        if rc not in (0, 3):
            bad.append(i)

    print("\n" + "=" * 70)
    if bad:
        print(f"结论：可见卡 {bad} **压测失败** —— 疑似硬件问题。")
        print("     临时规避：CUDA_VISIBLE_DEVICES 里把它们去掉，并让运维查 Xid / 换卡。")
        print("     查看内核日志： dmesg -T | grep -i xid | tail -30")
        sys.exit(1)
    print("结论：所有被检查的卡均通过压测。")
    print("    若训练仍中途失败，请转查：主机内存(swap)、/dev/shm、磁盘余量、")
    print("    以及带上堆栈重跑训练（torchrun --log-dir ... --tee 3）。")


if __name__ == "__main__":
    main()
