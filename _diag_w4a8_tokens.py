"""[TEMP DEBUG] W4A8_MXFP A5 诊断 v4：子进程网格扫描 (shape x token数 M)。

在 A5 上跑：python _diag_w4a8_tokens.py

e2e 定位到崩在 decode（M=1 单 token）的 FP4 matmul；之前诊断只测 M=128/16（prefill 尺寸）。
本脚本对每个 (shape, M) 组合**单独 fork 一个子进程**跑 1 次 matmul+sync，
段错误(-11)只杀子进程不影响整张表 —— 最后打印哪个组合 OK / SEGV / ERR。
调试完随其余 W4A8 debug 插桩一起删除。
"""

import sys

BLK = 32
SHAPES = [
    ("qkv_proj", 6144, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up_proj", 24576, 4096),
    ("down_proj", 4096, 12288),
]
MS = [1, 2, 3, 4, 6, 8, 16, 32, 128]


def worker(out, in_, m):
    import torch
    import torch_npu

    dev = "npu:0"
    try:
        torch_npu.npu.config.allow_internal_format = True  # 会被打回 False，正常
    except Exception:  # noqa: BLE001
        pass

    # NZ weight: packed-FP4 uint8 [out, in//2] -> cast29 -> transpose [in//2, out]
    w = torch.randint(0, 255, (out, in_ // 2), dtype=torch.uint8, device=dev)
    w = torch_npu.npu_format_cast(
        w, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    ).transpose(-1, -2)
    # weight_scale: [out, in//32] -> [in//64, out, 2]
    s = torch.randint(0, 255, (out, in_ // BLK), dtype=torch.uint8, device=dev)
    n, k = s.shape
    wscale = s.reshape(n, k // 2, 2).transpose(-3, -2)

    x = torch.randn(m, in_, dtype=torch.bfloat16, device=dev)
    qx, dscale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    out_t = torch_npu.npu_quant_matmul(
        qx,
        w,
        wscale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=dscale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=None,
        output_dtype=torch.bfloat16,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        group_sizes=[0, 0, BLK],
    )
    torch.npu.synchronize()  # 强制异步错误当场暴露
    assert out_t.shape == (m, out)
    print("WORKER_OK", flush=True)


def driver():
    import subprocess

    print("=" * 86, flush=True)
    print("W4A8 FP4 matmul 网格 (shape x M)  —  OK / SEGV<rc> / ERR", flush=True)
    print("=" * 86, flush=True)
    print("shape".ljust(14) + "".join(f"M={m}".rjust(8) for m in MS), flush=True)
    for name, out, in_ in SHAPES:
        row = name.ljust(14)
        for m in MS:
            r = subprocess.run(
                [sys.executable, __file__, str(out), str(in_), str(m)],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0 and "WORKER_OK" in r.stdout:
                cell = "OK"
            elif r.returncode < 0:
                cell = f"SEGV{r.returncode}"
            else:
                cell = "ERR"
            row += cell.rjust(8)
        print(row, flush=True)
    print("=" * 86, flush=True)
    print("看哪些 (shape,M) 是 SEGV：判断是 M=1 通杀，还是 shape+M 组合相关，", flush=True)
    print("以及最小安全 M（决定 apply 里 pad 到多少）。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        worker(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        driver()
