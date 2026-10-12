# GLM 显存建模公式与边界

本文记录工具当前对 `GLM-5.2`、`GLM-5.3` 普通版（`glm_moe_dsa`）实际执行的公式。它既是实现说明，也是回归审计基线。`GLM-5.3-Flash` 属于 KDA/MHC/ViT 独立架构，当前不能套用本文公式。

所有显存结果使用 GiB：

```text
GiB(x bytes) = x / 2^30
```

## 1. 符号

| 符号 | 含义 |
|---|---|
| `V` | vocabulary size |
| `H` | hidden size |
| `L` | 正常 decoder 层数，不含 MTP 层 |
| `m` | MTP 结构层数 |
| `E` | routed expert 数 |
| `Id` | dense MLP intermediate size |
| `Ie` | 单个 routed expert intermediate size |
| `Is` | 全部 shared expert 的合计 intermediate size |
| `nq` | attention head 数 |
| `rq`, `rkv` | Q/KV LoRA rank，属于 MLA 架构参数，不是训练 LoRA rank |
| `dn`, `dr`, `dv` | qk-nope、qk-rope、value head dim |
| `ni`, `di` | DSA index head 数、index head dim |
| `P` | FSDP rank/GPU 总数 |
| `TP`, `PP`, `EP`, `ETP` | tensor、pipeline、expert、expert-tensor 并行度 |
| `T` | 当前 microbatch token 数 |
| `b_art` | 冻结 checkpoint artifact 的平均存储字节/参数 |
| `r` | adapter 参数比例 |

## 2. GLM 结构参数量

### 2.1 顶层参数

```text
Embedding = V × H
LMHead    = 0              if tied embeddings
          = V × H          otherwise
FinalNorm = H
```

### 2.2 每层 MLA

```text
MLA =
    H × rq
  + rq
  + rq × nq × (dn + dr)
  + H × (rkv + dr)
  + rkv
  + rkv × nq × (dn + dv)
  + nq × dv × H
```

若配置开启 attention bias，再加：

```text
MLABias = rq + (rkv + dr) + H
```

### 2.3 每份 full DSA indexer

```text
Indexer = rq × ni × di + H × di + H × ni + 2 × di
```

标记为 `shared` 的层复用 indexer，不重复增加 indexer 参数。

### 2.4 MLP、router 与 norm

```text
DenseMLP      = 3 × H × Id
RoutedExperts = E × 3 × H × Ie
Router        = E × H
SharedExperts = 3 × H × Is
RouterBuffer  = E                 # fp32 buffer，不训练
DecoderNorm   = 2 × H
```

### 2.5 MTP 专属参数

除一份完整 sparse decoder 层外，每个 MTP 层还增加：

```text
MTPExtra = 2 × H² + 3 × H
```

其中 `2H²` 是 `eh_proj`，`3H` 是 `enorm/hnorm/shared_head.norm`。

### 2.6 总参数账本

默认规则中，前 3 层是 dense，之后是 sparse；前 3 层各有 full indexer，之后每 4 层再新增一份：

```text
D(L) = min(3, L)
S(L) = max(L - 3, 0)
F(L) = min(3, L) + floor(max(L - 3, 0) / 4)
```

令 `A = MLA + MLABias`，则：

```text
N = Embedding + LMHead + FinalNorm
  + (L + m) × A
  + (F(L) + m) × Indexer
  + D(L) × DenseMLP
  + (S(L) + m) × (RoutedExperts + Router + SharedExperts + RouterBuffer)
  + (L + m) × DecoderNorm
  + m × MTPExtra
```

`N` 包含非训练的 `RouterBuffer` 元素。checkpoint 字节不能一律写成 `N×2`：训练参数通常按 BF16 计 2 B，而该 buffer 按 FP32 计 4 B。

当前默认 78 层配置的账本为：

```text
N_total     = 753,329,940,480
N_trainable = 753,329,921,024
N_buffer    =              19,456
```

## 3. 减层是不是线性的

不是全局线性，而是“固定项 + 分段线性项 + 阶梯项”：

```text
N(L) = N_fixed
     + (L + m) × (MLA + DecoderNorm)
     + D(L) × DenseMLP
     + (S(L) + m) × SparseMLP
     + (F(L) + m) × Indexer
     + m × MTPExtra
```

- MLA 与 decoder norm 随 `L` 线性变化。
- `D(L)`、`S(L)` 在第 3 层处分段；减少 sparse 层和减少 dense 层的斜率不同。
- `F(L)` 每跨过 4 个 sparse 层才变化一次，是阶梯函数。
- embedding、lm head、final norm 与正常层数无关。
- 只修改 `L` 时，MTP 结构 `m` 默认保留，因此 MTP 是固定项。
- CE workspace、HCCL 与运行时预留不随 `L` 缩放。
- 未开 recompute 时，当前激活下界近似随 `L` 线性；开 full recompute 时工具把该项记为 0，并明确标为下界。

因此 8L/16L 是合成减层配置，不是新的模型种类，也不能用 `8/78`、`16/78` 直接缩放整卡显存。减层后没有官方 checkpoint 字节真值。

## 4. Checkpoint artifact 与训练精度

冻结基座使用：

```text
BaseWeightBytes = N_frozen × b_art
```

当前格式级估算：

| Artifact | `b_art` (B/param) | 说明 |
|---|---:|---|
| BF16 | 2 | 普通 BF16 |
| FP8 | 1.015625 | 1 B 数据 + 平均 scale 开销 |
| W8A8(C8) | 1.015625 | 格式级估算 |
| W4A8(C8) | 0.515625 | 0.5 B 数据 + 平均 scale 开销 |
| W4A4C8 MXFP4 | 0.53125 | 格式级估算 |
| W8A8C8 MXFP8 | 1.03125 | 格式级估算 |

这些低比特值不是逐 tensor inventory。能否被某个训练框架/kernel 直接加载，以及 QLoRA 反量化 workspace，都需要实测。

全参预训练、全参 SFT、RL actor update 会按训练 dtype 展开：

```text
FullUpdateWeightBytes = N_trainable_resident × 2 + N_fp32_buffer × 4
```

所以选择 W4/W8/FP8 artifact 不会把全参训练权重错误地按 4/8 bit 常驻。

## 5. Megatron 静态显存

### 5.1 组件到单 rank 的切分

每个组件 `c` 的单 rank 参数量为 `n_c × scale(c)`：

```text
scale(expert)     = 1 / (EP × ETP × PP)
scale(tp)         = 1 / (TP × PP)
scale(replicated) = 1 / PP
```

- `expert`：routed experts。
- `tp`：lm head、MLA q_b/kv_b/o、dense/shared MLP、MTP `eh_proj`。
- `replicated`：embedding、MLA q_a/kv_a、DSA indexer、router、norm、buffer。

`replicated` 只表示不按 TP/EP 切，当前仍按 `PP` 平均。`PP>1` 没有刻画首尾 stage、MTP placement 和层大小不均，所以是 stage-average 下界。

### 5.2 全参模式

对单 rank 上的常驻参数 `Nrank`，以及其中可训练的 `Ntrain`：

```text
Weight = Σ(trainable component rank × 2)
       + Σ(fp32 buffer rank × 4)

Grad   = Ntrain × 2
```

优化器在设备上时：

```text
Muon + AdamW = Nmuon × 4 + Nadamw × 12
纯 AdamW     = Ntrain × 12
```

其中二维权重按 Muon fp32 momentum 4 B 估算；embedding、bias、norm 等小参数按 AdamW 12 B 估算。选择 optimizer offload 时设备侧优化器项记为 0。

```text
Static = Weight + Grad + Optimizer
```

预训练、全参 SFT、RL actor update 的参数状态公式相同；RL 当前只计 actor，不含 reference、reward、critic、rollout engine 与 rollout KV cache。

### 5.3 LoRA / QLoRA

若用户给出精确 adapter 参数量：

```text
Nadapter = min(Neligible, Nadapter_user)
```

否则：

```text
Nadapter = Neligible × r
```

工具把同一个 `r` 均匀分配到可训练组件，这不是按真实 `target_modules` 逐层构造矩阵的精确账本。

```text
Weight = Nfrozen_rank × b_art
       + Nadapter_rank × 2
       + Nbuffer_rank × buffer_dtype_bytes

Grad      = Nadapter_rank × 2
Optimizer = Nadapter_rank × 12
```

LoRA 通常用 BF16 冻结基座；QLoRA 使用低比特冻结基座。adapter 本身仍按训练 dtype 计算。

## 6. Megatron 动态显存与整卡

```text
T    = micro_batch_size × (prompt_tokens + response_tokens)
Vloc = V / TP
```

输出阶段：

```text
CE          = 2 × T × Vloc × 4 / ce_chunks
Entropy     = entropy_enabled ? T × Vloc × 2 : 0
OutputPhase = CE + Entropy
```

GLM attention/路由阶段：

```text
DSAIndices    = T × index_topk × 4
RouterLogits  = router_fp32 ? T × E × 4 : 0
AttentionPhase = max(DSAIndices, RouterLogits)
Dynamic        = max(OutputPhase, AttentionPhase)
```

这里按阶段峰值取 `max`，不是把不会同时存活的 workspace 全部相加。

当前激活下界：

```text
Activation = 0                 if full recompute
           = L × T × H × 2     otherwise
```

通信与运行时：

```text
ncomm = 1
      + [TP > 1]
      + [PP > 1]
      + [EP > 1]
      + [ETP > 1]

HCCL    = ncomm × HCCL_BUFFSIZE_MB × 2 / 1024   # GiB
Runtime = HCCL + Fragment + PTA + DeviceOS
```

整卡：

```text
Peak   = Static + Dynamic + Activation
Device = Peak + Runtime
```

## 7. FSDP 静态显存

当前 FSDP 页固定 `EP=1`、`CP=1`，所有组件先统一按 `1/P` 分片。它不是 TP/PP/EP 与 FSDP 混合并行模型。

全参 GLM：

```text
Weight    = Σ(component bytes) / P
Grad      = Ntrain × 4 / P
Optimizer = Ntrain × 12 / P
Static    = Weight + Optimizer
```

FSDP 的 4 B gradient/reduce 输入与 Megatron 页的 2 B gradient 口径不同。为保持原 Qwen 页面黄金行为，Qwen FSDP 继续使用 8 B/param 优化器；GLM FSDP 使用 12 B/param。

LoRA/QLoRA 没有精确 `target_modules` 时按 LoRA rank 做方阵近似：

```text
r ≈ clamp(2 × lora_rank / H, 0.0001, 1)
```

然后：

```text
Weight    = (Nfrozen × b_art + Nadapter × 2 + buffer_bytes) / P
Grad      = Nadapter × 4 / P
Optimizer = Nadapter × 12 / P
```

## 8. FSDP 峰值

GLM 的一层近似工作单元：

```text
LayerUnit = MLA_per_layer
          + Indexer_per_full_layer
          + max(SparseMLP_per_layer, DenseMLP_per_layer)
          + 2H

UnitNumel = LayerUnit          if wrap whole layer
          = RoutedExpertUnit   otherwise

U = UnitNumel × base_resident_bytes / EP
```

动态项：

```text
AllGather   = U
CopyOut     = U
Prefetch    = prefetch_count × U
ReduceInput = UnitNumel × trainable_unit_ratio × 4
GMM         = U  if full update + grouped expert + nested wrap
              0  otherwise

Dynamic = AllGather + CopyOut + Prefetch + ReduceInput + GMM
```

```text
trainable_unit_ratio = 1   for full update
                     = r   for LoRA/QLoRA
```

激活下界：

```text
live_layers = 1          if recompute or activation offload
            = max(L, 1)  otherwise

Bfwd       = chunk_mbs if chunk_mbs > 0 else micro_batch_size
tokens     = Bfwd × sequence_length / CP
HiddenAct  = tokens × H × 2
Activation = live_layers × HiddenAct + GDN_skip + MoE_dispatch
```

当前 GLM 固定 `EP=CP=1` 且没有 GDN，实际退化为：

```text
Activation = live_layers × Bfwd × sequence_length × H × 2
```

通信与整卡：

```text
ncomm  = 1 + [EP > 1] + [CP > 1]
HCCL   = ncomm × HCCL_BUFFSIZE_MB × 2 / 1024

Peak   = Static + Grad + Dynamic + Activation
Device = Peak + HCCL + Fragment + PTA + DeviceOS
```

## 9. MTP 对显存的影响

必须区分两个动作。

### 9.1 给模型结构增加一个 MTP 层

从 `m=0` 改为 `m=1` 时：

```text
ΔNmtp = MLA + MLABias
      + Indexer
      + RoutedExperts + Router + SharedExperts + RouterBuffer
      + DecoderNorm
      + 2H² + 3H
```

默认 GLM-5.2/5.3 普通架构中：

```text
ΔNmtp_total     = 9,952,920,576
ΔNmtp_trainable = 9,952,920,320
ΔNmtp_buffer    =           256
BF16 resident   ≈ 18.539 GiB globally
```

之后再按 Megatron 的组件切分或 FSDP 的 `1/P` 分片。这个动作会增加 checkpoint/常驻权重，也会增加训练状态和真实前后向 workspace。

### 9.2 模型已有 MTP，只开/关辅助训练目标

工具的开关语义是“这些现有参数是否更新”，不是“从 checkpoint 删除 MTP”：

```text
ΔWeight(OFF → ON) = 0
```

全参 Megatron：

```text
ΔGrad = MTP_trainable_rank × 2

Muon + AdamW:
ΔOpt  = MTP_muon_rank × 4 + MTP_adamw_rank × 12

纯 AdamW:
ΔOpt  = MTP_trainable_rank × 12
```

若优化器 offload，则设备侧 `ΔOpt=0`。

FSDP：

```text
ΔGrad = ΔNmtp_trainable × 4 / P
ΔOpt  = ΔNmtp_trainable × 12 / P
```

LoRA/QLoRA 在按比例 `r` 估算时：

```text
Megatron:
ΔAdapterWeight = r × MTP_trainable_rank × 2
ΔGrad          = r × MTP_trainable_rank × 2
ΔOpt           = r × MTP_trainable_rank × 12

FSDP:
上式的 MTP 参数先除 P，且 ΔGrad 改为 4 B/adapter param
```

若用户固定的是“精确 adapter 总参数量”而不是比例，打开 MTP 不一定增加 adapter 总量，只会改变统一分摊比例。

以 Megatron `TP=4, PP=8, EP=32, ETP=1`、Muon+AdamW、优化器不 offload 为例，现有 MTP 从不训练改为训练：

```text
MTP resident weight（开关前后都有） ≈ 0.092 GiB/rank
新增 gradient                      ≈ 0.092 GiB/rank
新增 optimizer                     ≈ 0.184 GiB/rank
设备静态增量                       ≈ 0.276 GiB/rank
```

### 9.3 当前尚未计入的 MTP 增量

当前实现中：

```text
ΔActivation_from_MTP_toggle = 0
ΔDynamic_from_MTP_toggle    = 0
```

这只是工具当前的下界，不代表真实 MTP 训练没有运行时增量。真实训练还会增加 MTP 辅助层的 forward/backward 保存张量、辅助 logits/loss、调度中同时存活的张量，以及可能的通信和 kernel workspace。没有目标框架的执行图、microbatch schedule 和实测 profile 时，不把这些项伪装成精确值。

## 10. 已知下界与不支持项

- full recompute 的激活不可能真正为 0；当前 0 表示“未建模的下界”。
- 未计 MLA/DSA backward 临时张量、MoE dispatch/sort、FlashAttention、量化反量化 kernel workspace。
- Megatron 未计 pipeline 同时在途 microbatch，`PP>1` 仅按 stage 平均。
- FSDP 的 whole-layer/expert unit 是近似峰值，不是 allocator trace。
- LoRA 参数按比例或 `2r/H` 近似；拿到真实 `target_modules` 后应改用精确 adapter 参数清单。
- RL 仅含 actor update。
- artifact 名称中的 `MTP` 只是 checkpoint 元数据，不能代替独立的 MTP 训练开关。
- `GLM-5.3-Flash` 未启用；不能用普通 GLM-5.3 的 MLA/DSA/MoE 账本估算。

实现位置：[`glm-modeling.js`](glm-modeling.js)、[`training-policies.js`](training-policies.js)、[`megatron.html`](megatron.html)、[`fsdp.html`](fsdp.html)。
