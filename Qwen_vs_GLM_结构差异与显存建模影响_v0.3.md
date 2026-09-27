# Qwen 与 GLM 结构差异及其对显存建模的影响（v0.3）

> 任务：WEIY-17（父任务 WEIY-8 显存建模工具）　日期：2026-09-24　版本：v0.3（v0.2 复核勘误版，取代同日 v0.2 与 09-22 v0.1）
> 基准代码：`qwen-tools-glmdev/memory/megatron.html`（`parseConfig` L445、`countParams` L485、`compute` L557）
> 证据来源：ModelScope 各 checkpoint 的 `config.json`、`model.safetensors.index.json`、safetensors 分片 header（真实 dtype/shape）、repo 文件清单（分片字节数求和）。逐条标注「事实 / 推断 / 待验证」。

---

## 0a. v0.3 更新摘要（独立复核 + 勘误）

对 v0.2 全部关键事实做了第二轮独立复核（config / index.json / 分片 header 重新拉取，逐项重算参数量与字节数）。**结论：v0.2 的结构与参数量主体结论全部成立**（§2 六个模型的参数量、分片数、字节数、indexer 共享分布、MTP 两族结构、层布局、KDA/MHC/ViT 张量清单、代码行为断言均复现，误差 <0.05%）。发现并修正 3 处错误、2 处表述不清：

1. **FP8/BF16 边界修正（GLM-5.3 与 5.3-Flash 的口径不同，且与 v0.2 §4.8 表述有出入）**——经分片 header 逐张量核验 + 字节数收支互证（§4.8 新表）：
   - **GLM-5.3**：`kv_b_proj` 与 indexer 的 `wk/wq_b` **是 F8_E4M3，不是 BF16**（v0.2 §1 行 10 的「kv_b/indexer 仍 bf16」对 5.3 不成立）。5.3 的 BF16 集合 = embed / lm_head / 各 norm / router gate / `indexer.weights_proj` / `eh_proj`；F32 = `e_score_correction_bias` 与 `weight_scale_inv`。
   - **GLM-5.3-Flash**：BF16 范围比 v0.2 写的更大——**KDA 注意力全套投影（q/k/v/o/b/g_a/g_b/f_a/f_b/conv1d，≈4.68B 参数）也是 BF16**；MLA 五项中只有 **`kv_b_proj` 是 BF16，`q_a/q_b/kv_a/o` 均为 F8**。FP8 集合 = experts + shared_experts + dense MLP（gate/up/down）+ MLA 的 q_a/q_b/kv_a/o。字节数收支：314.39GB(F8) + 13.84GB(BF16) + 0.08GB(scale/f32) ≈ **328.31GB，与实际 total_size 328.33GB 吻合（差 0.02GB）**；5.3 同法得 755.63GB vs 实际 755.62GB。
2. **4.7-Flash 词表更正**：V=**154880**（与 5.x 同，非 4.x 的 151552）——影响 CE 峰值项 `2·T·(V/TP)·4` 的取值（v0.2 §1 行 14 的「4.x 151552」仅对 4/4.5/4.6/4.7 成立）。
3. **4.7-Flash 配置补全**（使 31.22B 可复现）：L=**47**（+MTP=48 个 MLA 层）、dense 层 intermediate_size=**10240**、nKV=20（MLA 名义值）。v0.2 §1 行 1 的「48 层」即 47+MTP，此处明确。
4. 表述澄清：KDA 每层参数 **137.7M**（§4.5 张量清单）**+ MHC 0.79M = 138.5M**（v0.2 §1 行 6 的「138.5M + hc 0.79M」写法易读成重复相加）。
5. `modules_to_not_convert` 的**幽灵模块名是普遍现象**：5.3-Flash 的列表还引用了不存在的 `qkv_proj`/`fused_qkvbfg_a_proj`/`attn_mha`/`attn_mqa`/`hyper_connection`/`mapping_proj`（实际张量是分离的 `q_proj/k_proj/v_proj` 等）——**dtype 判定必须以 safetensors header 为准**，该列表只能作旁证（v0.2 已发现 `indexers_proj`，此为其推广）。

---

## 0b. v0.2 更新摘要

v0.1 覆盖了 GLM-4-9B / 4.5 / 4.5-Air / 4.6 / 5，但**任务要求的 GLM5 系列（glm4.7、5.1、5.2、5.3 各参数量档）未覆盖**。v0.2 补齐并修正：

**新增覆盖（均为事实，证据见 §2/§9）**：GLM-4.7、GLM-4.7-Flash、GLM-5.1、GLM-5.2、GLM-5.3、GLM-5.3-Flash（共 4 个参数量档：31B / 321B / 358B / 753B）。

**对 v0.1 的修正**：

1. **GLM-4.5/4.7/4.7-Flash 的 MTP 层自带独立 `embed_tokens`（V×H）**——v0.1 的张量清单漏了它。GLM-4.7 的 MTP 实际 ≈ **5.54B**，不是 v0.1 的 3.99B，更不是工具公式 的 ≈3.87B。
2. **`shared_head.head` 是独立张量（[V,H]），与 `lm_head` 不共享**（至少 released checkpoint 如此；独立序列化即非 tie）。v0.1「待验证 #1」对 glm4_moe 族解决；**glm5 族（5/5.1/5.2/5.3/5.3-Flash）的 MTP 没有独立 head/embed，只有 `shared_head.norm`**——两族 MTP 结构不同，需分支处理。
3. **Qwen 基线更正**：工具现役模型是 Qwen3.5 / 3.6 / 3.8（全部为 GDN 线性注意力 + 每 4 层 full attention 的混合架构，MoE 或 dense），v0.1 用 Qwen3 系对比。结构轴结论不变，但「Qwen 已有功能」清单应以 3.5/3.6/3.8 为准——其中 `layer_types`/`full_attention_interval`/GDN 公式**已存在**，是覆盖 GLM-5.3-Flash 混合层的现成资产。
4. **`model.safetensors.index.json` 的 `metadata.total_size` 不可靠**：GLM-4.5/4.7 报 358.34 GB（≈参数量而非 bf16 字节数，真实 716.68 GB）；GLM-5.x 又是字节数。参数量核验一律用 repo 文件清单求和。
5. GLM-5.1 经张量级复核 ≡ GLM-5，v0.1 全部结论直接适用。

---

## 1. 结论速览

以现有工具逻辑处理 GLM，错误分两层：**解析层**（config key 不匹配 → MoE 被当 dense）和**公式层**（key 映射后各结构项仍算错）。

| # | 差异项 | 涉及 GLM | 工具现状 | 影响 | 量级 | 证据 |
|---|---|---|---|---|---|---|
| 0 | **config key 不匹配**（`n_routed_experts`/`num_nextn_predict_layers`/…） | 全部 | `parseConfig` 只认 Qwen key | **E→0、mtp→0：整个 MoE 不可见** | GLM-4.7 被算成 ≈31.5B dense 模型，真实 358.3B（**11× 低估**）；所有 GLM MoE 同理 | 事实（config + 代码） |
| 1 | MLA 注意力（q/kv LoRA + 独立 v_head） | 4.7-Flash / 5 / 5.1 / 5.2 / 5.3 / 5.3-Flash | 只有 GQA/GDN | 参数误算（方向随版本）+ KV 布局不同 | 5.1：单层 165M vs 误算 100M，79 层**少算 ~5.1B**；5.2/5.3 误算 302M/层反而**多算 ~10.8B**；4.7-Flash：head_dim 缺失→工具猜 H/nQ，48 层少算 ~0.24B | 事实（tensor shape） |
| 2 | DSA indexer | 5 / 5.1（每层）、5.2 / 5.3 / 5.3-Flash（部分层） | 无 | 参数/激活漏算 | 5.1：79 层 ≈ **0.74B** 全漏 | 事实（tensor shape） |
| 2b | **indexer 跨层共享（`indexer_types` full/shared，周期 4）** | 5.2 / 5.3 / 5.3-Flash | 无 | 5.2 只有 **22/79 层**有 indexer 权重（21 regular + MTP） | 权重 0.74B→**0.21B**（−72%）；若把 5.1 的每层插件直接套 5.2，**多算 0.53B**；index cache / top-k 激活按 4 层一组折减（推断，`index_topk_freq=4`） | 事实（权重分布）+ 推断（激活） |
| 3 | first_k_dense_replace（前 k 层 dense） | 4.5 / 4.5-Air / 4.6 / **4.7** / 5 系全部 | 假设全层 MoE | 专家高估、dense 漏算 | 4.7：净多算 **~10.8B**（同 4.5）；5.1/5.2/5.3：净多算 **~28.4B**（3×9.70B − 0.68B）；5.3-Flash：~21.4B | 事实（tensor shape） |
| 4 | MTP = 完整层 + eh_proj +（glm4 族：独立 embed + head） | 全部 GLM MoE | 半层注意力估算 | 低估 | 4.7：工具 3.87B vs 实际 **5.54B**（少 1.67B；其中 embed+head 2×V×H = 1.55B）；5.2：完整层 9.95B vs 半层口径少 ~0.09B | 事实（tensor shape） |
| 5 | **MLA-nope 变体（`qk_rope_head_dim=0`）** | 5.3-Flash | 无 | kv_a 无 rope 分量（[512,H] 而非 [576,H]）、qk_nope=256 | glm_mla 插件必须参数化，不能写死 GLM-5 的 192/64/576 | 事实（tensor shape） |
| 6 | **KDA 线性注意力（gated delta，15 种张量/层）** | 5.3-Flash（34/45 层） | GDN 公式张量集不同（无 b_proj/g_a/g_b/f_a/f_b/A_log/dt_bias/3×conv1d） | 公式不匹配 | 真实 ≈137.7M/层（+MHC 0.79M 合计 138.5M）；按 GDN 公式喂同维度 ≈168.4M/层，34 层**多算 ~1.0B**（且张量清单错；`qkv_proj`/`fused_qkvbfg_a_proj` 为 config 幽灵名，实际是分离张量） | 事实（tensor shape）+ 算术 |
| 7 | **MHC 多路超连接（`mhc:true`, `hc_mult=4`, sinkhorn）** | 5.3-Flash | 无 | 参数小、**激活大**：残差流疑似 ×4 | 参数仅 2×(24×4H+27)≈0.79M/层（全模型 36M）；但激活公式 `L×T×H×2` 可能需乘 ~4（推断，待 profile） | 事实（tensor shape）+ 推断（激活） |
| 8 | **indexer kpool 压缩（`index_kpool=4`）** | 5.3-Flash | 无 | 新张量 + 索引布局变化 | `compress_ape [4,128]` + `compress_gate [128,H]`（+0.52M/层）；index cache 布局待验证 | 事实（tensor shape） |
| 9 | **ViT 视觉塔（多模态）** | 5.3-Flash | 有 Qwen3-VL ViT 公式（近似） | 结构相近但有差项 | ≈0.56B：多 downsample conv [4096,1024,2,2]、attn q/k norm、MLP 带 bias、merger SwiGLU+proj+post_norm；激活随图像 token 数 | 事实（tensor shape） |
| 10 | **FP8 默认 checkpoint（e4m3 + block 128×128 + `weight_scale_inv`）** | 5.3（默认）/ 5.3-Flash | 单一 `bw` 字节宽 | 混合精度 per-module（两模型边界不同，见 §4.8 表） | 5.3：755.62GB fp8 vs 5.2 同构 1506.67GB bf16；5.3-Flash：328.33GB 混合。**5.3 的 kv_b/indexer 主权重为 F8**；**5.3-Flash 的 KDA 注意力全套/kv_b/indexer/MHC/ViT 为 BF16**（v0.2 此处口径有误，已按 header 勘误）；训练口径（bf16 从零 vs fp8 起步）待定 | 事实（header dtype + 字节收支互证） |
| 11 | QKV bias | 4.x / 4.5 / 4.5-Air / 4.6 / **4.7**（4.7-Flash 与 5.x 无） | 无 bias 项 | 低估（小） | 4.7：14336/层 × 93 ≈ 1.3M | 事实（tensor shape） |
| 12 | partial RoPE / rope_interleave / qk_norm 开关 | 各异（4.7-Flash `partial_rotary_factor=1.0` 全 RoPE） | 无此概念 | **无显存影响**（逐元素） | 0；仅需解析记录 | 事实（config） |
| 13 | `moe_router_dtype: float32` + `e_score_correction_bias`(fp32) | 5.2 / 5.3 / 5.3-Flash | 未计 | 激活小项、精确对齐项 | router logits `T×E` fp32；bias E×4B | 事实（config/shape） |
| 14 | vocab 差异 | GLM-5.x 与 4.7-Flash 均 154880 / 4、4.5、4.6、4.7 为 151552 vs Qwen3.8 248320 | 公式可复用 | CE 峰值项 `2·T·(V/TP)·4` | GLM 每 token CE 比 Qwen3.8 小 ~38%（口径不变）；4.7-Flash 实为 154880（v0.2 曾误记 151552，已勘误） | 事实（config） |

**一句话**：glm4.7 = glm4.5 同构（直接复用 v0.1 结论）；**4.7-Flash 引入「MLA 无 indexer」新组合**；**5.1 = 5**；**5.2/5.3 = 5.1 + indexer 跨层共享（−72% indexer 权重）**；**5.3-Flash 是全新家族**（KDA 线性注意力 + MLA-nope + DSA + kpool + MHC + ViT + FP8），其混合层模式与 Qwen3.5/3.8 同构、可部分复用工具已有的 `layer_types`/GDN 资产，但张量级公式全部需要新写。

---

## 2. 目标系列盘点（GLM5 系列：glm4.7、5.1、5.2、5.3 各参数量档）

参数量 = 解析核对值（与 repo 分片字节求和互证，误差 <0.01%）。均为事实。

| 模型 | architecture | 参数量 | 注意力 | MoE | MTP | 与既有结论关系 |
|---|---|---|---|---|---|---|
| GLM-4.7 | `Glm4MoeForCausalLM` | **358.3B**（716.68GB bf16，93 分片） | GQA 96/8×128 + QKV bias + qk_norm | E=160, I=1536, shared=1, topk=8, **first_k=3** | 1（完整层 + 独立 embed + head） | **≡ GLM-4.5 结构**，v0.1 §3.4/3.7/3.8 直接适用（另加本文 MTP 修正） |
| GLM-4.7-Flash | `Glm4MoeLiteForCausalLM` | **31.2B**（62.44GB bf16，48 分片；H=2048，L=**47**+MTP，V=**154880**，dense I=**10240**） | **MLA（q_lora=768, kv_lora=512, nope=192, rope=64, v=256），无 indexer** | E=64, I=1536, shared=1, topk=4, **first_k=1**, noaux_tc, scale=1.8 | 1（完整层 + 独立 embed + head） | **新组合**：glm_mla 插件的「无 indexer」参数化用例 |
| GLM-5.1 | `GlmMoeDsaForCausalLM` | **753.9B**（1507.73GB bf16，282 分片） | MLA + DSA indexer **每层** | E=256, I=2048, shared=1, topk=8, first_k=3, sigmoid/noaux_tc | 1（完整层 + indexer，无 head/embed） | **≡ GLM-5 结构**，v0.1 §3.1/3.2 直接适用 |
| GLM-5.2 | `GlmMoeDsaForCausalLM` | **753.3B**（1506.67GB bf16，282 分片） | MLA 每层 + **indexer 仅 22/79 层**（full: 0,1,2,6,10,…,74 + MTP；其余 shared） | 同 5.1 + **`moe_router_dtype=fp32`**、显式 `mlp_layer_types` | 1（完整层 + indexer） | 5.1 − indexer 共享（−0.53B）+ router fp32 |
| GLM-5.3 | `GlmMoeDsaForCausalLM` | **753.3B**（fp8 755.62GB，141 分片） | = 5.2 | = 5.2 | = 5.2 | **config 与 5.2 仅差 `quantization_config`（FP8 默认）**；另有 BF16 版独立 repo（`ZhipuAI/GLM-5.3-BF16`，已核实存在） |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` | **≈321B**（fp8 混合 328.33GB，62 分片；LM 320.8B + ViT ≈0.56B） | **混合**：34 层 KDA 线性 + 11 层 MLA-nope+DSA(+kpool) + MTP=DSA 层 | E=288, I=2048, shared=1, topk=8, first_k=3, router fp32 | 1（完整 DSA+MoE 层） | **全新家族**，见 §5 |

> 官方未发布 GLM-5.x-Air 档；「各参数量档」目前由 4.7-Flash（31B）/ 5.3-Flash（321B）/ 4.7（358B）/ 5.x（753B）构成。（事实）

**Qwen 基线（工具现役，事实）**：Qwen3.8-2.4T-A95B（H=8192, L=92, E=512, I=2048, topk=10, shared=2048, GQA 64/4×256 + GDN 混合 interval=4, MTP=1, V=248320）；Qwen3.5-397B-A17B / Qwen3.6-35B-A3B / Qwen3.8-27B 同族（全 GDN+full 混合、全层 MoE 或全 dense、无 QKV bias、无 dense/MoE 混层、无 MLA/DSA/indexer、无 indexer 共享）。Qwen 侧无 FP8 默认、无 ViT 的 Flash 混合档。

---

## 3. 解析层失败：直接喂 GLM config 会发生什么（事实 + 代码对读）

`parseConfig`（megatron.html L445）的 pick 列表 vs GLM key：

| 工具读的 key | GLM 实际 key | 后果 |
|---|---|---|
| `num_experts` / `num_local_experts` | `n_routed_experts` | **E=0 → MoE 模型被当 dense**，`moeI=0`，EP/ETP UI 隐藏 |
| `mtp_num_hidden_layers` | `num_nextn_predict_layers` | **mtp=0 → MTP 全漏** |
| `shared_expert_intermediate_size` | `n_shared_experts × moe_intermediate_size` | shared=0 |
| `intermediate_size`（dense 用） | ✅ 同名，但 GLM 只有前 k 层用 | 会被当全层 dense MLP |
| `head_dim` | 5.1=64、5.2/5.3=192（均非 MLA 真实维度）、5.3-Flash=0；**4.7-Flash 无此字段** | 4.7-Flash 触发 `H/nQ` 回退（=102.4，错） |
| `layer_types` / `linear_*`（GDN） | 5.3-Flash 有 `layer_types`（三态）+ `linear_attn_config`（嵌套 dict） | linear 参数读不到 → hasGdn=false → 全按 full GQA |
| — | `first_k_dense_replace` / `mlp_layer_types` / `indexer_types` / `q_lora_rank`… | 全部无对应 |

净效果（算术）：GLM-4.7 config → 工具算出 ≈**31.5B** dense 模型（92 层 × (denseMLP 0.189B + GQA 0.136B) + embed 1.55B），真实 **358.3B**。**任何公式级修正之前必须先做 key 映射**（P0）。

---

## 4. 公式层差异（逐项量化，key 映射后仍会算错的部分）

### 4.1 MLA（4.7-Flash / 5.x / 5.3-Flash 的 DSA 层）

单层参数（事实，shape 核对）：

- 5.1/5.2/5.3：`q_a[H,2048] + q_b[16384,2048] + kv_a[576,H] + kv_b[28672,512] + o[16384→H]` ≈ **165M/层**；工具 GQA 口径误读方向随子版本不同：5.1 的 config `head_dim=64` → ≈100M/层，79 层**少算 5.1B**（v0.1 结论，5.1 复核成立）；**5.2/5.3 的 config `head_dim=192`** → ≈302M/层，79 层反而**多算 ~10.8B**——同一误读逻辑在不同 GLM 版本上误差方向相反，靠调系数无法对齐。
- 4.7-Flash：`q_a[768,2048] + q_b[5120,768] + kv_a[576,2048] + kv_b[8960,512] + o[2048,5120]`（o 为 [H, nQ×v_head]） ≈ **21.8M/层**；工具 head_dim 缺失回退 H/nQ=102.4 → ≈16.8M/层 → 48 个 MLA 层（47+MTP）**少算 0.24B**。
- 5.3-Flash（nope 变体）：`q_a[1536,4096] + q_b[16384,1536] + kv_a[512,4096]（无 rope 分量）+ kv_b[32768,512] + o[4096,16384]` ≈ **117.5M/层**（仅 11+1 层）。

建模要求：`glm_mla(H, q_lora, kv_lora, nope, rope, v_head, nQ)` 全参数化；`rope=0` 合法；indexer 开关独立（4.7-Flash = off）。推理 KV：latent = `kv_lora + rope`（5.3-Flash 为 512，5.x 为 576）。

### 4.2 DSA indexer 与跨层共享（新）

- 5.1：indexer（`wk[128,H] + wq_b[4096,2048] + weights_proj[32,H] + k_norm[128]×2`）≈ 9.4M/层 × **79 层** = 0.74B（事实）。
- 5.2/5.3：`indexer_types` = 21 full + 57 shared（周期 4：full 层 0,1,2,6,10,…,74），**权重仅存在于 full 层 + MTP（22 层）** = 0.21B（事实，index.json 张量分布核对）。`index_topk_freq=4`、`index_skip_topk_offset=3`、`index_share_for_mtp_iteration=true` → 推断：top-k 索引每 4 层计算一次、组内共享，indexer 前向/反向激活与推理 index cache 也应按组折减（**待验证**：训练实现是否如此）。
- 5.3-Flash：indexer 仅在 12 层（11 DSA + MTP），新增 kpool 张量：`compress_ape[4,128] + compress_gate[128,H]`（+0.52M/层）。`index_kpool=4, index_kpool_compress=true, index_kpool_always_select_tail=true` → 推断 key 池化 4×，影响索引分数计算量与 cache 布局（待验证）。
- 若把「每层 indexer」的 glm5 插件直接套 5.2/5.3：**多算 57 × 9.4M ≈ 0.53B**。

### 4.3 dense/MoE 混层（first_k_dense_replace / mlp_layer_types）

| 模型 | naive 全层 MoE | 正确 | 净差 |
|---|---|---|---|
| 4.7（=4.5） | 92 层 × 3.80B | 89 MoE + 3 dense(0.57B) + MTP MoE | **+10.8B 多算** |
| 5.1/5.2/5.3 | 78 层 × 9.70B | 75 MoE + 3 dense(0.68B) + MTP MoE | **+28.4B 多算** |
| 5.3-Flash | 45 层 × 7.27B | 42 MoE + 3 dense(0.45B) + MTP MoE | **+21.4B 多算** |

（事实 + 算术；经 `N_expert/(EP×ETP×PP)` 分片后按 rank 缩放，若 Adam 在 GPU 误差 ×~8。）5.2+ 同时给出显式 `mlp_layer_types` 数组，建议解析它而非只认 `first_k_dense_replace`（更通用，二者并存时以数组为准）。

### 4.4 MTP（两族结构，事实）

| 族 | MTP 组成 | 实际参数 | 工具公式 | 差 |
|---|---|---|---|---|
| glm4_moe（4.5/4.6/**4.7**）| 完整层（GQA+MoE）+ eh_proj[2H→H] + **独立 embed_tokens[V,H] + shared_head.head[V,H]** + 3 norm | 4.7：**5.54B** | ≈3.87B（半注意力、无 eh_proj/embed/head） | **−1.67B** |
| glm4_moe_lite（**4.7-Flash**） | 完整层（MLA+MoE）+ 同上 | 1.28B | 同上口径 | −0.66B（embed+head 0.63B + eh_proj 8.4M + …） |
| glm_moe_dsa（5/5.1/5.2/5.3） | 完整层（MLA+indexer+MoE）+ eh_proj + shared_head.norm（**无独立 embed/head**） | 5.2：9.95B | 半层口径 ≈9.87B | −0.09B |
| glm5_next（**5.3-Flash**） | 完整 DSA 层（MLA-nope+indexer+kpool+MoE）+ eh_proj + shared_head.norm | ≈7.4B | — | 需新公式 |

修正项：MTP = 完整层 + eh_proj(2H·H) + norms + `glm4 族另加 2×V×H`。v0.1 待验证 #1 就此解决（独立张量，非 tie；训练框架是否再共享梯度属低概率，保留观察项）。

### 4.5 KDA 线性注意力（5.3-Flash，34/45 层；事实）

每层张量与参数（H=4096, nH=64, d=128）：

```
q/k/v_proj [8192,4096]×3 = 100.7M   o_proj [4096,8192] = 33.6M
b_proj [64,4096] = 0.26M            g_a[128,4096]+g_b[8192,128] = 1.57M（门控低秩）
f_a[128,4096]+f_b[8192,128] = 1.57M（遗忘低秩，推断命名语义）
q/k/v_conv1d [8192,1,4]×3 = 0.10M   dt_bias(f32)[8192]  A_log(f32)[64]  o_norm[128]
合计 ≈ 137.7M/层
```

工具 GDN 公式（Qwen3.5 资产）张量集不同（单一 ba_proj、无 b/f 门控低秩、单 conv）；按同维度代入得 ≈168.4M/层，34 层差 ~1.0B 且张量清单错——**必须新写 `glm5_kda` 公式**，不能复用 gdn。KDA 训练激活：状态量与 T 无关（线性注意力），但 chunk 式并行实现的中间量待 profile（待验证）。`short_conv_kernel_size=4`、`gate_lower_bound=-5` 为数值细节，不影响显存形状。

### 4.6 MHC 多路超连接（5.3-Flash；事实 + 推断）

每层 6 张量：`hc_attn/ffn_base(f32)[24]`、`hc_attn/ffn_fn(bf16)[24, 4H]`、`hc_attn/ffn_scale(f32)[3]` ≈ 0.79M/层（全模型 36M，参数可忽略但须计入精确对齐）。`hc_mult=4` + `hc_sinkhorn_iters=20`：推断残差流按 4 路并行扩展（`fn` 的 4H 维度），**重计算关闭时激活 ≈ L×T×H×2×~4**——对 `L×T×H×2` 公式是一个潜在的大乘子（**待验证**：需官方实现或 profile）。

### 4.7 ViT（5.3-Flash；事实）

24 blocks（Hv=1024, Iv=4096）：attn qkv[3072,1024]+proj+**q_norm/k_norm**；MLP **带 bias**；`patch_embed` conv3d[1024,3,2,14,14]；**`downsample` conv[4096,1024,2,2]**（Qwen 无）；merger：SwiGLU gate/up[10240,4096]+down+proj[4096,4096]+post_projection_norm。合计 ≈**0.56B**。工具 `vitBlock/vitMerger` 公式（Qwen3-VL 口径）缺 downsample/qk-norm/MLP bias/merger 结构——参数差 ~3%（推断），激活侧图像 token 数（448²/14²/merge²/downsample²）待定口径。

### 4.8 FP8 与精度（5.3 / 5.3-Flash；事实，v0.3 经 header 逐张量核验 + 字节收支互证）

两模型均为 e4m3 + `weight_block_size[128,128]` + 每量化张量配 `weight_scale_inv`（fp32，shape = ceil(dim/128)），但 **BF16 保留集差异很大，不能共用一张映射表**：

| 模块 | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| experts（gate/up/down）、shared_experts | F8 | F8 |
| dense MLP（gate/up/down） | F8 | F8 |
| MLA `q_a/q_b/kv_a/o_proj` | F8 | **F8** |
| MLA `kv_b_proj` | **F8**（v0.2 误记 BF16，已勘误） | **BF16** |
| KDA 注意力全套（q/k/v/o/b/g_a/g_b/f_a/f_b/conv1d） | — | **BF16**（≈4.68B 参数，v0.2 未计入保留集，已勘误） |
| indexer（wk/wq_b） | **F8**（v0.2 误记 BF16，已勘误） | BF16 |
| indexer（weights_proj/k_norm/kpool_ape/kpool_gate） | BF16 | BF16 |
| MHC（hc_*：base/scale=F32，fn=BF16） | — | F32/BF16 |
| router gate / e_score_correction_bias | BF16 / F32 | BF16 / F32 |
| embed / lm_head / 各 norm / eh_proj / ViT 全部 | BF16 | BF16 |

**字节收支互证（事实 + 算术）**：
- GLM-5.3：F8 参数 ≈751.24B（751.24GB）+ BF16 ≈2.10B（4.21GB）+ scale/bias ≈0.18GB ≈ **755.63GB**，实际 `total_size` 755.62GB（差 0.02GB，<0.01%）。
- GLM-5.3-Flash：F8 ≈314.39B（314.39GB）+ BF16 ≈6.92B（13.84GB）+ scale/f32 ≈0.08GB ≈ **328.31GB**，实际 328.33GB（差 0.02GB）。

对建模：(a) 推理/加载口径 bw 需 per-module 且 5.3 与 5.3-Flash 用不同映射；(b) 预训练从零通常仍 bf16（推断，待训练配置确认）；(c) `quantization_config.modules_to_not_convert` **含大量幽灵模块名**（5.3 的 `indexers_proj`；5.3-Flash 的 `qkv_proj`/`fused_qkvbfg_a_proj`/`attn_mha`/`attn_mqa`/`hyper_connection`/`mapping_proj` 均无对应张量）——**dtype 判定一律以 safetensors header 为准**，该列表只作旁证。

### 4.9 其余（沿用 v0.1，适用范围更新）

- QKV bias：4.x/4.5/4.6/**4.7** 有（14336/层 × 93 ≈ 1.3M）；**4.7-Flash 与 5.x 全系无**（attention_bias=false）。
- partial RoPE：4.7 = 0.5；**4.7-Flash = 1.0（全 RoPE）**；5.x 走 MLA 的 rope 分量。显存影响 0。
- MoE 口径（gate_up=2HI、down=IH、router=EH、shared=3H·shareI）：全系可复用，仅 key 映射；`topk_method=noaux_tc`/sigmoid 路由激活 `T×E`（5.2+ fp32）。
- 并行切分：`N_dense/(TP×PP) + N_expert/(EP×ETP×PP)` 可复用；MLA 的 q_a/kv_a 低秩维切分、KDA 的 head 维切分约束待实现确认（推断）；混层导致 PP stage 负载不均（首 stage 多 dense 层）。

---

## 5. GLM-5.3-Flash 层布局（事实，index.json + config 交叉核对）

```
层 0,1,2   : KDA 线性注意力 + dense MLP(12288)
层 3       : MLA-nope + DSA indexer + kpool + sparse MoE(E=288)
层 4,5,6   : KDA + sparse MoE          ← 每 4 层一个 DSA 层（3,7,11,…,43）
…
层 43      : DSA + MoE
层 44      : KDA + MoE
层 45(MTP) : DSA(MLA-nope+indexer) + MoE + eh_proj + shared_head.norm
每层       : MHC(hc_attn_* + hc_ffn_*)
ViT        : 24 blocks + patch_embed + downsample + merger
```

`layer_types` 三态（`linear_attention` / `deepseek_sparse_attention`），与 Qwen3.5 的二态（`full_attention`/`linear_attention`）同机制不同取值——工具 `layerTypes` 数组机制可直接扩展（`nFull = filter(t => t.includes("full"))` 的匹配逻辑需改为显式枚举）。

---

## 6. 显存峰值阶段（预训练 vs SFT）

| 阶段 | 结论 | 依据 |
|---|---|---|
| 前向激活 | 4.7/4.7-Flash：与 Qwen 同族（full recompute → 0）。5.1：MLA `q_b/kv_b` 中间张量 + indexer `T×T` 风险（v0.1）。**5.2/5.3：indexer 共享使 indexer 激活按 1/4 组折减（推断）**。5.3-Flash：**KDA 层激活与 T 解耦（线性）**，但 **MHC ×4 残差流**可能使非重计算激活接近 4×；DSA 层同 5.x 风险 | 推断，全部待 profile |
| CE/logits | `2·T·(V/TP)·4`：V=154880（5.x）/151552（4.7）vs Qwen 248320——公式复用，GLM 每 token 更小 | 事实 |
| 反向 | MLA 激活保存量大（v0.1）；MoE dispatch/combine `T×topk×H×2` 未计（v0.1 P1，维持）；KDA 反向的 delta-rule 重放待验证 | 推断 |
| 优化器步 | Adam 12B/参数：参数量误差直接 ×12 放大（Adam 在 GPU）；4.7 的 MTP embed+head 1.55B → GPU Adam 下 +18.6GB | 事实+算术 |
| SFT 特有 | MTP 是否参与 SFT 待确认；FP8 checkpoint 起步口径待确认；5.3-Flash 的 ViT 是否训练待确认；SFT 序列短 → CE/静态占比更大 | 待验证 |

---

## 7. 需新增/修改的建模项（可直接排期）

| 优先级 | 建模项 | 类型 | 覆盖 |
|---|---|---|---|
| **P0** | GLM config 解析映射（§3 全表 + `indexer_types`/`mlp_layer_types`/`linear_attn_config`/`quantization_config`） | 修改 | 全部 |
| **P0** | `first_k_dense_replace`/`mlp_layer_types` 分层 | 修改 | 4.5/4.7/5.x/5.3-Flash |
| **P0** | MTP 完整层 + eh_proj + 两族分支（glm4 族 +2×V×H embed/head；glm5 族仅 norm） | 修改 | 全部 MoE GLM |
| **P0** | `glm_mla` 参数化插件（q_lora/kv_lora/nope/rope/v_head/nQ；rope=0；indexer 开关） | 新增 | 4.7-Flash/5.x/5.3-Flash |
| **P0** | `glm_dsa_indexer` 插件（+ `indexer_types` 共享感知 + kpool 张量） | 新增 | 5.x/5.3-Flash |
| **P0** | `glm5_kda` 线性注意力插件（15 张量公式） | 新增 | 5.3-Flash |
| **P0** | `layer_types` 三态枚举（弃用 `includes("full")` 隐式匹配） | 修改 | 5.3-Flash（兼容 Qwen 二态） |
| **P0** | QKV bias 开关 | 修改 | 4.x/4.5/4.7 |
| P1 | MHC 激活乘子（hc_mult）建模 | 新增 | 5.3-Flash |
| P1 | MLA/KDA/indexer 的 TP/EP 切分约束校验 | 新增 | 5.x/5.3-Flash |
| P1 | MoE dispatch/combine 激活、router fp32 logits | 修改 | MoE 全部 |
| P1 | ViT 公式补差（downsample/qk-norm/bias/merger）+ 图像 token 口径 | 修改 | 5.3-Flash |
| P1 | FP8 混合精度 per-module bw（5.3 与 5.3-Flash 边界不同，按 §4.8 表） | 新增 | 5.3/5.3-Flash |
| P1 | 推理 KV/index cache（MLA latent + `index_topk`/freq/kpool 折减） | 修改 | 5.x/5.3-Flash |
| P2 | 旧 `ChatGLMModel` 命名映射；`e_score_correction_bias`；忽略 config 幽灵项（`indexers_proj`） | 修改 | 边缘 |

**对齐验证基准（建议直接做单测）**：countParams 应复现 4.7=358.34B、4.7-Flash=31.22B、5.1=753.86B、5.2/5.3=753.34B、5.3-Flash LM≈320.76B + ViT≈0.56B（本文 §2，与分片字节和互证 <0.01%）。

---

## 8. 建议的 GLM 支持范围（供选型决策）

| 插件 | 覆盖 | 复用 | 工作量 |
|---|---|---|---|
| `glm4_dense` | GLM-4-9B-0414 | GQA+SwiGLU 大部分 | 小 |
| `glm4_moe_gqa` | 4.5/4.5-Air/4.6/**4.7** | GQA+共享专家+MoE | 中（混层、MTP 两族、QKV bias、config 映射） |
| `glm_mla`（参数化） | **4.7-Flash**（indexer=off）/ 5 / 5.1 / 5.2 / 5.3 | MoE/并行/Norm | 大（MLA 五项、indexer 开关与共享） |
| `glm5_hybrid` | **5.3-Flash** | `layer_types` 机制、ViT 框架 | 最大（KDA、MLA-nope、kpool、MHC、FP8、ViT） |

**MVP（09-27 原型）建议**：`glm4_dense` + `glm4_moe_gqa`（用 4.5-Air 对齐，4.7 同构免测）+ `glm_mla` 先覆盖 **4.7-Flash**（31B，checkpoint 仅 62GB，参数对齐成本最低，且逼出插件参数化）；5.x 全系与 5.3-Flash 单列迭代，**必须先拿真实 profile** 校准 MLA/DSA/KDA/MHC 激活，否则只能给参数静态下界。

---

## 9. 未确认假设与待验证清单

1. **5.2/5.3 indexer 共享的训练实现**：top-k 是否每 4 层物化一次（`index_topk_freq=4`）；index cache 折减系数。
2. **kpool 语义**（5.3-Flash）：key 池化 4× 后 index_topk=2048 的有效口径。
3. **MHC 激活放大**：残差流是否 ×4（hc_mult）；sinkhorn 是否引入额外中间量。
4. **KDA 训练激活**：chunk 式 delta-rule 的中间张量规模；KDA 的 TP 切分维度。
5. **MLA 并行切分**：q_lora/kv_lora 在 TP/EP 下的切分与通信量（v0.1 遗留，5.x 复核仍无官方配置）。
6. **官方训练配置/技术报告**：GLM-5.x 系列均未获取；Megatron/MindSpeed 切分、recompute 粒度、Adam offload。
7. **MTP 训练口径**：glm4 族独立 embed/head 是否在训练中共享梯度；SFT 是否启用 MTP。
8. **FP8 checkpoint 的 SFT/继续预训练口径**（5.3/5.3-Flash）：checkpoint 内 F8/BF16 边界已按 header 确证（§4.8），但训练起步精度（是否反量化到 bf16 / 保 F8 训练）仍待确认。
9. **5.3-Flash 视觉部分**：是否参与训练；图像 token 数口径（448/14/merge2/downsample2）。
10. **真值口径**（WEIY-8 未决）：「真实训练显存」测量来源与 <1% 阈值的判定方式。
11. GLM-4.6/4.7 与 4.5 的权重级 diff（config 仅 max_position_embeddings 不同；结构结论已由张量清单确认一致，低优先）。

---

## 10. 来源

**ModelScope（`https://modelscope.cn/models/<repo>/resolve/master/...`；v0.2 于 2026-09-24 拉取，v0.3 同日独立复核）**

- `config.json`：`ZhipuAI/GLM-4.7`、`ZhipuAI/GLM-4.7-Flash`、`ZhipuAI/GLM-5.1`、`ZhipuAI/GLM-5.2`、`ZhipuAI/GLM-5.3`、`ZhipuAI/GLM-5.3-BF16`（存在，BF16 无量化）、`ZhipuAI/GLM-5.3-Flash`、`ZhipuAI/GLM-4.5`、`ZhipuAI/GLM-4.6`、`ZhipuAI/GLM-4-9B-0414`；`Qwen/Qwen3.8-2.4T-A95B`、`Qwen/Qwen3.8-27B`、`Qwen/Qwen3.6-35B-A3B`、`Qwen/Qwen3.5-397B-A17B`
- `model.safetensors.index.json` + 分片 safetensors header（Range 请求解析 dtype/shape）：上述 GLM 全部（v0.3 复核：5.2/5.1/5.3/5.3-Flash/4.7/4.7-Flash 的 index + 5.2/5.3/5.3-Flash/4.7/4.7-Flash 共 9 个分片的 header，含 F8/BF16/F32 dtype 与全部关键 shape）
- index `total_size` 实测：4.7=358,337,791,296（=参数量 358.34B，非字节数）、4.7-Flash=31,221,488,576（=参数量 31.22B）、5.1=1,507,728,316,928（1507.73GB）、5.2=1,506,659,919,872（1506.66GB）、5.3=755,617,140,416（fp8）、5.3-Flash=328,326,771,576（fp8 混合）；分片数 93/48/282/282/141/62 全部核对
- 张量分布实测：5.2 indexer 张量仅存在于层 [0,1,2,6,10,…,74,78]（22 层）；5.1 indexer 全部 79 层；5.3-Flash MLA/indexer 层 [3,7,…,43,45]、KDA 层 34 个、dense 层 [0,1,2]、MHC 45 层（不含 MTP）
- 参数量交叉验证：解析公式计数 vs 张量字节，误差 <0.01%（§2 表、§4.8 字节收支）

**代码**：`qwen-tools-glmdev/memory/megatron.html`（`QWEN_GROUPS` L348、`parseConfig` L445、`countParams` L485、`compute` L557）

**前版**：v0.2（2026-09-24，附件 `WEIY-17_Qwen_vs_GLM_结构差异与显存建模影响_v0.2.md`）——补齐 GLM5 系列覆盖；v0.3 在其基础上独立复核并勘误（§0a）。v0.1（2026-09-22）——GLM-4.5/4.5-Air/4.6/5 与 Qwen3 的对比、MLA/DSA/MoE 公式推导仍有效。

**未获取**：GLM-5.x 官方技术报告、官方训练并行配置、真实训练显存 profile。
