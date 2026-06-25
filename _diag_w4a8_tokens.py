"""[TEMP DEBUG] W4A8_MXFP A5 诊断 v5：主进程内降序扫 M，定位 decode 小 M 阈值。

在 A5 上跑（sglang/qwen3_dense_w4a8/ 目录下）：python _diag_w4a8_tokens.py

v4 的子进程网格在这环境里每个 worker 都 HANG（torch_npu 在 fork 出来的裸子进程里
卡在设备初始化，跟矩阵乘无关），已废弃。本版回到**主进程**直跑（与早先能跑通的
diag_w4a8_shapes.py 同款），对 o_proj 形状 (4096,4096) 按 M **从大到小** 扫描：
能跑的大 M 先逐个打印 OK，跑到某个小 M 卡住(HANG)就停 —— 最后一个打印的
`trying M=X` 后面没有 `OK M=X`，那个 X 就是会挂的最大 M，它上面那个 OK 的 M
就是「最小安全 M」（= apply 里要 pad 到的阈值）。

卡住后直接 Ctrl-C 即可，把已打印的行贴回来。
"""

import torch
import torch_npu

DEV = "npu:0"
BLK = 32

# 确认会崩的 o_proj；如需对比再加 ("qkv_proj", 6144, 4096)
SHAPES = [
    ("o_proj", 4096, 4096),
]
# 降序：大 M 先过，卡在边界停
MS = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]


def build_nz_weight(out, in_):
    w = torch.randint(0, 255, (out, in_ // 2), dtype=torch.uint8, device=DEV)
    w = torch_npu.npu_format_cast(
        w, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    ).transpose(-1, -2)
    s = torch.randint(0, 255, (out, in_ // BLK), dtype=torch.uint8, device=DEV)
    n, k = s.shape
    wscale = s.reshape(n, k // 2, 2).transpose(-3, -2)
    return w, wscale


print("=" * 72, flush=True)
print("torch", torch.__version__, "| torch_npu", torch_npu.__version__, flush=True)
try:
    torch_npu.npu.config.allow_internal_format = True  # 会被打回 False，正常
except Exception as e:  # noqa: BLE001
    print("set allow_internal_format:", e, flush=True)
print("=" * 72, flush=True)

for name, out, in_ in SHAPES:
    print(f"\n### {name}  OUT={out} IN={in_}  (M 降序，卡住即为阈值边界)", flush=True)
    weight, wscale = build_nz_weight(out, in_)
    for m in MS:
        x = torch.randn(m, in_, dtype=torch.bfloat16, device=DEV)
        qx, dscale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        print(f"  trying M={m} ...", flush=True)  # 卡住时这是最后一行
        try:
            out_t = torch_npu.npu_quant_matmul(
                qx, weight, wscale,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale=dscale, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                bias=None, output_dtype=torch.bfloat16,
                x2_dtype=torch_npu.float4_e2m1fn_x2, group_sizes=[0, 0, BLK],
            )
            torch.npu.synchronize()
            print(f"  OK M={m} out={tuple(out_t.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR M={m}: {e}", flush=True)

print("\n" + "=" * 72, flush=True)
print("最后一个 OK 的 M = 最小安全阈值；它下面 trying 后没 OK 的 M 会 HANG。", flush=True)
print("=" * 72, flush=True)
