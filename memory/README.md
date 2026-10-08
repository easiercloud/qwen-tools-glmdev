# 显存分析

单卡显存估算：Megatron 页支持 Qwen 及 GLM-5.2/5.3，FSDP 与 MoE 推理页保持原有能力。

在线：

- [Megatron 显存建模](https://zyang6.github.io/qwen-tools/memory/megatron.html)
- [FSDP 显存建模](https://zyang6.github.io/qwen-tools/memory/fsdp.html)
- [推理显存建模](https://zyang6.github.io/qwen-tools/memory/infer.html)

源码目录即本文件夹。GitHub：[zyang6/qwen-tools](https://github.com/zyang6/qwen-tools)。

## 选哪一页

| 页面 | 场景 | 静态怎么切 | 高峰 |
|---|---|---|---|
| [`megatron.html`](megatron.html) | Megatron actor update | `N_rank × 4`（bf16 权重+梯度；Adam 在 CPU 则为 0） | 词表 CE：`2×T×(V/TP)×4` |
| [`fsdp.html`](fsdp.html) | FSDP 全量微调 | `N/P × 10`（不含梯度） | 梯度×2 + `6U` + 激活 |
| [`infer.html`](infer.html) | MoE 推理部署 | 权重按 EP/TP 切 | KV / 激活 / HCCL / ACLGraph / CANN / OS |

Megatron 的 N 不按卡数 P 均分，按 TP/PP/EP/ETP。FSDP 才是 `÷P`。推理页嵌入 [Eco-Sphere Memory Planner](https://eco-sphere.github.io/infer-memory-visualizer/)，权重从 safetensors header 逐张量求和。

## 使用方法

1. 从总入口打开对应页面，或直接进上面的链接。
2. 先选模型系列，再选模型。
3. 改切分、序列、offload / recompute，右侧实时出单卡 GiB（`1024³`）。
4. Dense 没有专家：EP=1。GLM 默认 TP=8、EP=32；这是可修改的工具建议值，不是官方训练拓扑。

## GLM-5.2 / 5.3 

- 5.2、普通 5.3、5.3-BF16 使用同一 `glm_moe_dsa` 组件账本：3 个 dense 层、75 个 sparse 层、1 个 MTP 参数层，DSA indexer 共 22 份。
- 5.2 BF16 checkpoint 账本与官方索引字节精确相等；普通 5.3 的 FP8 checkpoint 字节与 BF16 训练权重口径分开显示。
- GLM 默认优化器为 Muon + AdamW 小参数；Muon 的 4 B/二维参数状态是待目标训练实现校准的估算口径。
- CE 分块、DSA top-k indices 和 fp32 router logits 单独展示；不同执行阶段的 workspace 取最大值，不直接相加。
- PP>1 当前仍是按 stage 均摊的下界；整卡峰值在取得真实训练 profile 前明确标为下界。
- GLM-5.3-Flash 是 KDA/MHC/ViT 独立架构，目录中可见但当前禁用，不能套用普通 5.3 公式。
- 减层验证：目录预置 `GLM-5.2 (8L)/(16L)` 合成条目，GLM 模型可直接改"层数（减层覆盖）"输入框，或粘贴任意 config.json；dense 前 3 层与 indexer full/shared 频率按官方规则重新推导，合成配置无 checkpoint 真值，不再与官方字节对账。

页面右上角 **GitHub** 看源码，**反馈** 开 Issue（标题已带 `[memory/megatron]` 或 `[memory/fsdp]`）。推理页的按钮指向 Eco-Sphere 仓库。

## Megatron 口径

```text
N        = Text + ViT
N_expert = L × (gate_up + down)     # dense 为 0
N_dense  = N − N_expert
N_rank   = N_dense/(TP×PP) + N_expert/(EP×ETP×PP)
静态     = N_rank × 4 / 1024³       # Adam CPU → 优化器 0
T        = mbs × (prompt + response)
动态内存 = 2 × T × (V/TP) × 4 / 1024³
激活值   = 0                         # full recompute；关掉才 L×T×H×2
HCCL     = n_comm × BUFFSIZE × 2 / 1024
整卡     = 静态 + 动态 + 激活 + HCCL + 碎片1 + PTA1 + Device OS 3.5
```

`n_comm = 1(DP) + [TP>1] + [PP>1] + [EP>1] + [ETP>1]`。高峰不加 MoE permute。param/grad offload 只改步间，不改高峰。

## FSDP 口径

```text
静态 = N/P × 10          # bf16 参数+Adam；梯度不算进静态
高峰再叠：梯度×2 + 6U + 激活 + HCCL/碎片/PTA/OS
```

## 推理口径

嵌入 Eco-Sphere 工具，计算在对方页面完成：

```text
单卡 = 权重 + KV/Index Cache + 激活 + HCCL
     + ACLGraph + CANN/PTA/算子 + Device OS
```

说明与贡献见 [infer-memory-visualizer README](https://github.com/Eco-Sphere/infer-memory-visualizer/blob/main/README.md)。

## 本目录文件

```text
memory/
├── README.md        # 本文件
├── megatron.html    # Megatron actor update
├── model-catalog.js # 模型系列、模型、repo 与能力目录
├── glm-modeling.js  # GLM 配置解析、组件账本与纯计算
├── fsdp.html        # FSDP 全量微调
└── infer.html       # 推理（iframe）
```

## 反馈

发现公式、切分或模型解析有问题，请开 [Issue](https://github.com/zyang6/qwen-tools/issues/new)，附：

- 页面（Megatron / FSDP）
- 模型与切分
- 工具结果 / 预期 / 日志或 pickle 来源
