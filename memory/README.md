# 显存分析

单卡显存估算：Megatron 与 FSDP 页支持 Qwen 及 GLM-5.2/5.3，并统一支持预训练、全参 SFT、LoRA、QLoRA 和强化学习 actor update。

在线：

- [Megatron 显存建模](https://zyang6.github.io/qwen-tools/memory/megatron.html)
- [FSDP 显存建模](https://zyang6.github.io/qwen-tools/memory/fsdp.html)
- [推理显存建模](https://zyang6.github.io/qwen-tools/memory/infer.html)

源码目录即本文件夹。GitHub：[easiercloud/qwen-tools-glmdev](https://github.com/easiercloud/qwen-tools-glmdev)。

## 选哪一页

| 页面 | 场景 | 静态怎么切 | 高峰 |
|---|---|---|---|
| [`megatron.html`](megatron.html) | Megatron 预训练 / SFT / LoRA / QLoRA / actor update | 按 TP/PP/EP/ETP 和训练策略拆分 | 词表 CE、DSA/路由与激活下界 |
| [`fsdp.html`](fsdp.html) | FSDP 预训练 / SFT / LoRA / QLoRA / actor update | 按 FSDP 分片和训练策略拆分 | 梯度、all-gather、激活与运行时 |
| [`infer.html`](infer.html) | MoE 推理部署 | 权重按 EP/TP 切 | KV / 激活 / HCCL / ACLGraph / CANN / OS |

Megatron 的 N 不按卡数 P 均分，按 TP/PP/EP/ETP。FSDP 才是 `÷P`。推理页嵌入 [Eco-Sphere Memory Planner](https://eco-sphere.github.io/infer-memory-visualizer/)，权重从 safetensors header 逐张量求和。

## 使用方法

1. 从总入口打开对应页面，或直接进上面的链接。
2. 先选模型系列和基础模型 release，再选 checkpoint/artifact；BF16、FP8、W4/W8 不再伪装成不同架构。
3. 选择 GLM 时再选 checkpoint/artifact 与训练方式；Megatron LoRA/QLoRA 可填 adapter 参数比例或精确参数量，FSDP 按 LoRA rank 估算。
4. 改切分、序列、offload / recompute，右侧实时出单卡 GiB（`1024³`）。
5. Dense 没有专家：EP=1。Megatron 的 GLM 建议初值为 TP=4、PP=8、EP=32；这是可修改的工具建议值，不是官方训练拓扑。

## GLM-5.2 / 5.3 

- 5.2 和普通 5.3 是基础模型 release；BF16、FP8、W8A8(C8)、W4A8(C8)、MXFP4、MXFP8 是独立的 checkpoint/artifact 选择。
- 5.2/5.3 使用同一 `glm_moe_dsa` 组件账本：3 个 dense 层、75 个 sparse 层、1 个 MTP 参数层，DSA indexer 共 22 份。
- 5.2 BF16、5.2 FP8、5.3 BF16 和 5.3 FP8 已记录精确 checkpoint 总字节；低比特内部版本若没有索引真值，会明确显示“无精确真值”。
- GLM 默认优化器为 Muon + AdamW 小参数；Muon 的 4 B/二维参数状态是待目标训练实现校准的估算口径。
- CE 分块、DSA top-k indices 和 fp32 router logits 单独展示；不同执行阶段的 workspace 取最大值，不直接相加。
- PP>1 当前仍是按 stage 均摊的下界；整卡峰值在取得真实训练 profile 前明确标为下界。
- GLM-5.3-Flash 是 KDA/MHC/ViT 独立架构，目录中可见但当前禁用，不能套用普通 5.3 公式。
- 减层验证：8L/16L 不是模型种类。直接修改“层数（减层覆盖）”输入框即可；dense 前 3 层与 indexer full/shared 频率按官方规则重新推导，合成配置无 checkpoint 真值。
- MTP 参数是否存在来自架构 config；“训练 MTP 辅助目标”是独立开关。artifact 名称里的 `MTP` 只作为原始元数据记录，不会偷偷打开训练开关。
- 完整参数量、减层、Megatron/FSDP、LoRA/QLoRA 与 MTP 增量公式见 [`GLM_FORMULAS.md`](GLM_FORMULAS.md)。

## Qwen 正交兼容

- 无 URL 参数时仍默认进入 Qwen3.5-27B；Qwen 原有模型选项、控件可见性和计算口径保持不变。
- GLM 的 artifact、训练模式、减层、MTP 和 Muon 控件只在 GLM 模式显示，不改变 Qwen 的原界面。
- Qwen FSDP 继续使用原有 8 B/param 优化器状态口径；GLM 使用 12 B/param，二者由显式模型分支隔离。
- 回归测试固定 Qwen 默认 Megatron/FSDP 的完整显存明细，避免共享模块以后反向污染 Qwen。

## 训练策略口径

- 预训练、全参 SFT、强化学习 actor update：权重按训练 dtype 常驻，量化 artifact 只表示加载来源，不会把全参训练权重直接按 4/8 bit 计算。
- LoRA：冻结基座，只给 adapter 分配训练权重、梯度和优化器状态。
- QLoRA：量化冻结基座常驻，adapter 仍按训练 dtype 计算；量化 scale 的平均开销包含在 artifact 的 `storageBytesPerParam` 中。
- 强化学习当前只覆盖 actor update，不包含 reference、reward、critic 与 rollout engine。

页面右上角 **GitHub** 看源码，**反馈** 开 Issue（标题已带 `[memory/megatron]` 或 `[memory/fsdp]`）。推理页的按钮指向 Eco-Sphere 仓库。

## Megatron 口径

```text
N        = Text + ViT
N_expert = L × (gate_up + down)     # dense 为 0
N_dense  = N − N_expert
N_rank   = N_dense/(TP×PP) + N_expert/(EP×ETP×PP)
全参静态 = N_rank × (权重 + 梯度) + optimizer
LoRA静态 = 冻结基座 + adapter权重 + adapter梯度 + adapter optimizer
T        = mbs × (prompt + response)
动态内存 = 2 × T × (V/TP) × 4 / 1024³
激活值   = 0                         # full recompute；关掉才 L×T×H×2
HCCL     = n_comm × BUFFSIZE × 2 / 1024
整卡     = 静态 + 动态 + 激活 + HCCL + 碎片1 + PTA1 + Device OS 3.5
```

`n_comm = 1(DP) + [TP>1] + [PP>1] + [EP>1] + [ETP>1]`。高峰不加 MoE permute。param/grad offload 只改步间，不改高峰。

## FSDP 口径

```text
全参和 adapter 模式都按训练策略拆分 base / trainable / frozen，再应用 FSDP 分片。
高峰再叠：当前策略的梯度 + all-gather 工作集 + 激活 + HCCL/碎片/PTA/OS。
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
├── GLM_FORMULAS.md  # GLM 参数、训练状态、峰值与 MTP 公式
├── megatron.html    # Megatron actor update
├── model-catalog.js # 模型系列、模型、repo 与能力目录
├── training-policies.js # artifact、训练模式与可训练参数共享口径
├── glm-modeling.js  # GLM 配置解析、组件账本与纯计算
├── fsdp.html        # FSDP 多训练模式
└── infer.html       # 推理（iframe）
```

## 反馈

发现公式、切分或模型解析有问题，请开 [Issue](https://github.com/easiercloud/qwen-tools-glmdev/issues/new)，附：

- 页面（Megatron / FSDP）
- 模型与切分
- 工具结果 / 预期 / 日志或 pickle 来源
