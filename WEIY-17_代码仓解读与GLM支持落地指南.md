# WEIY-17 代码仓解读与 GLM 支持落地指南（移交用）

> 日期：2026-09-24　用途：交给实现 agent 的移交文档　配套：《WEIY-17_Qwen_vs_GLM_结构差异与显存建模影响_v0.4.md》（结构证据与公式，本文简称「对比文档 v0.4」）
> 实现优先级（用户已定）：**先 GLM-5.2，再 GLM-4.7**。5.3-Flash 不在本期范围。

---

## 1. 背景、目的与要达成的目标

### 1.1 项目背景（父任务 WEIY-8「显存建模工具」）

面向智谱 GLM 系列模型，建立 GPU/NPU **显存估算与方案对比工具**。产品定义见 `GLM显存建模工具_调研与需求方案.md`（v0.1，2026-09-21）：「GLM Memory Planner & Flight Recorder」双闭环——运行前规划（至少多少卡 / 可行并行方案 / 各方案显存吞吐对比）+ 运行中校准诊断。UI 概念见 `UI_SPEC.md`（MemScope）。**9 月 27 日前完成可评审原型**。

核心质量目标：**估算显存与真实训练显存差距 < 1%**。任何未确认假设必须显式标注（本指南沿用对比文档 v0.4 的「事实/推断/待验证」三级标注）。

### 1.2 现有基础

工具在 `qwen-tools-glmdev`（可运行的静态 HTML 工具台）上扩展，已支持 Qwen 全系（Qwen3/3.5/3.6/3.8、Qwen3-VL、Qwen2.5-VL）的 Megatron/FSDP 训练显存估算与 MoE 推理预算。**GLM 全系当前不可用**：config key 不匹配导致 MoE 被当 dense（GLM-4.7 会被算成 31.5B，真实 358.3B，11× 低估），公式层还有 MLA/DSA/MTP 等十余项差异（对比文档 v0.4 §1 全表）。

### 1.3 本期要达成的目标（验收口径）

1. **GLM-5.2**（744B-A40B，MLA + DSA indexer 跨层共享 + MoE + MTP）与 **GLM-4.7**（358B-A32B，GQA + 混层 + 两族 MTP）两个模型在 Megatron 页（`memory/megatron.html`）可用：选模型 → 拉真实 config → 出逐项显存分解。
2. **参数量对齐（硬验收）**：`countParams` 复现张量级真值（对比文档 v0.4 §2，与分片字节和互证 <0.01%）：
   - GLM-5.2 = **753.34B**（GLM-5.1 = 753.86B、GLM-5 = 753.86B 同构顺带）
   - GLM-4.7 = **358.34B**（GLM-4.5/4.6 同构）
   - 顺带目标（低成本）：4.7-Flash = 31.22B、5.3 = 753.34B（config 同 5.2）
3. **显存分解对齐**：权重/梯度按官方 TP/EP plan 切分（§3.4）；优化器按 **Muon 口径**（GLM 官方用 Muon 不是 Adam，v0.4 §4.10）；CE 峰值带 chunk 折减选项。
4. 结构证据全部来自 checkpoint 事实 + 官方代码（本文 §4 索引），不引入无来源假设。

---

## 2. 仓库解读（当前代码全貌）

### 2.1 两个仓库的关系

```
/Users/ywy/Documents/codeproject/显存建模工具/     ← 父仓（文档 + 工作树宿主）
├── GLM显存建模工具_调研与需求方案.md              ← 产品需求 v0.1（§1.1）
├── UI_SPEC.md                                     ← MemScope UI 概念
├── GLM_Memory_Planner_Meeting_Deck.pptx           ← 评审材料
├── WEIY-17_*.md (v0.2/v0.3)                       ← 对比文档历史版（v0.4 见本次附件）
├── qwen-tools-glmdev/                             ← 嵌套独立 git 仓（工具本体，main 跟踪 zyang6/qwen-tools 上游的 glmdev 副本）
└── .git/worktrees/…                               ← Multica agent 工作树（agent/agent/weiy-17-…，与 main 同步）
```

- 工具代码全部在 **qwen-tools-glmdev**（父仓不含代码；父仓 main 当前在 b266383，工作树分支上有 agent 的自动保存提交）。
- 本地预览：`cd qwen-tools-glmdev && python3 -m http.server 8876 --bind 127.0.0.1`。

### 2.2 工具模块布局（qwen-tools-glmdev）

```
index.html          总入口（GitHub Pages，模块导航）
memory/             01 显存分析（本期全部工作在此）
  megatron.html     ★ Megatron 训练显存（GLM 支持主战场，1012 行）
  fsdp.html         FSDP 全量微调（二期迁移 GLM）
  infer.html        推理（iframe 嵌 Eco-Sphere 工具，不改）
data/ perf/ ops/ issues/ reports/   空（预留）
```

### 2.3 megatron.html 架构（读懂这四个函数 = 读懂全页）

| 位置 | 函数 | 职责 | GLM 接入点 |
|---|---|---|---|
| L348 | `QWEN_GROUPS` + `repoForModel` | 模型下拉与 ModelScope repo 映射 | 新增 `GLM_GROUPS`（ZhipuAI/GLM-5.2、ZhipuAI/GLM-4.7…） |
| L445 | `parseConfig(cfg)` | ModelScope `config.json` → 内部模型对象（**只认 Qwen key**） | GLM key 映射（§3.2 全表） |
| L485 | `countParams(model)` | 参数量逐项分解（embed/lm/attention/MLP/MoE/GDN/MTP/ViT） | MLA/indexer/两族 MTP 公式（§3.3） |
| L557 | `compute(model, train)` | 切分（TP/PP/EP/ETP）→ 单卡 GiB 分解 + 峰值 | TP 复制项、Muon、CE chunk（§3.4） |

现有口径（`memory/README.md`）：`N_rank = N_dense/(TP×PP) + N_expert/(EP×ETP×PP)`；静态 = `N_rank × 4`（bf16 权重+梯度，Adam 在 CPU 则优化器 0）；Adam 在 GPU = `×12`；动态 = `2×T×(V/TP)×4`（CE）；激活 = full recompute ? 0 : `L×T×H×2`；HCCL = `n_comm×BUFFSIZE×2`；整卡加碎片 1G + PTA 1G + OS 3.5G。模型配置从 `https://modelscope.cn/models/<owner>/<name>/resolve/master/config.json` 拉取（L923）。另有三个本地预设（`local:12L/20L/27B`）绕过网络。

其他值得知道的：`defaultTrain`（L333，MoE 默认 TP=2/EP=8）、`syncSplitUi`（dense 隐藏 EP/ETP）、`partsOf/render`（右侧堆叠条渲染）、`applyParsedModel/loadWeight`（L943/955，config 应用与加载）。`fsdp.html` 为同构的另一套口径（`N/P×10`，高峰梯度×2 + 6U + 激活），本期不改但注意别在共享逻辑上做破坏性重构。

### 2.4 git 状态

qwen-tools-glmdev：`main` 干净（590ae42，上游 zyang6/qwen-tools 的 glmdev 演化线）。工作约定：单页单文件、无构建步骤、内联 JS/CSS，改动以页面为单位小步提交（看 git log 风格：`Split tools into modules and publish Megatron as memory/megatron.html`）。

---

## 3. 需求指南：GLM-5.2 → GLM-4.7 落地任务清单

> 依据：对比文档 v0.4 §3/§4/§7 + 官方代码事实。所有「事实」= checkpoint/官方代码可复验。

### 3.1 任务 A（P0）：模型清单与 GLM 预设

- `QWEN_GROUPS` 旁新增 `GLM_GROUPS`：`ZhipuAI/GLM-5.2`（先行）、`ZhipuAI/GLM-5.1`、`ZhipuAI/GLM-5`、`ZhipuAI/GLM-4.7`、`ZhipuAI/GLM-4.7-Flash`（可选：`ZhipuAI/GLM-5.3`，config 同 5.2）。`repoForModel` 加 `ZhipuAI/` 前缀分支。
- 注意 `index.html`（总入口）与 `memory/README.md` 的模块说明同步补 GLM。

### 3.2 任务 B（P0）：`parseConfig` 的 GLM key 映射

对照表（左=现状读的 key，右=GLM 实际；事实，v0.4 §3）：

| 内部字段 | 现读 key | GLM key（5.2 / 4.7） | 备注 |
|---|---|---|---|
| experts | `num_experts`/`num_local_experts` | **`n_routed_experts`**（transformers 亦映射 `num_experts`） | 5.2=256；4.7=160 |
| moeI | `moe_intermediate_size` | 同名 ✓ | |
| shareI | `shared_expert_intermediate_size` | **`n_shared_experts × moe_intermediate_size`** | 两模型均 1× |
| denseI | `intermediate_size` | 同名 ✓ | 仅前 3 层用（first_k） |
| mtp | `mtp_num_hidden_layers` | **`num_nextn_predict_layers`** | 均 1 |
| headDim | `head_dim` | **禁用**。读 `q_lora_rank/kv_lora_rank/qk_nope_head_dim/qk_rope_head_dim/v_head_dim`（MLA 模型）；GQA 模型（4.7）用 `head_dim`=128 | **官方加载逻辑强制 `head_dim=qk_rope_head_dim`，该字段不可信**（v0.4 §0-3） |
| — | 无 | **`first_k_dense_replace` / `mlp_layer_types`**（数组优先） | 混层 |
| — | 无 | **`indexer_types`**（数组；或 `index_topk_freq`+`index_skip_topk_offset` 推导：`full iff max(i−offset+1,0)%freq==0`） | 5.2：21 full + 57 shared |
| — | 无 | **`index_topk/index_head_dim/index_n_heads`** | 2048/128/32 |
| — | 无 | `moe_router_dtype`（5.2=fp32）、`topk_method=scoring_func` 组合 `noaux_tc`+`sigmoid`、`routed_scaling_factor` | 激活小项 |
| — | 无 | `attention_bias`（4.7=true，5.2=false）、`partial_rotary_factor`（4.7=0.5） | |
| — | 无 | `quantization_config`（5.3 才有，本期 5.2/4.7 为 bf16，先解析存储即可） | |

### 3.3 任务 C（P0）：`countParams` 的 GLM 公式

**(c1) MLA 注意力**（5.2；事实，官方形状核对）：
```
q_a = H×q_lora               q_b = nQ×(nope+rope)×q_lora
kv_a = (kv_lora+rope)×H      kv_b = nQ×(nope+v_head)×kv_lora
o   = nQ×v_head×H            + q_a_layernorm(kv_lora? no: q_lora) + kv_a_layernorm(kv_lora)
5.2 数值：H=6144, q_lora=2048, kv_lora=512, nope=192, rope=64, v_head=256, nQ=64 → ≈165.0M/层 × 79 层（78+MTP）
```
4.7 走现有 GQA 公式（96/8×128 + QKV bias + qk_norm）。

**(c2) DSA indexer**（5.2；事实）：
```
full 层：wk[128,H] + wq_b[nQ×nope? 实为 4096,2048] + weights_proj[32,H] + k_norm×2 ≈ 9.4M
仅 indexer_types[i]=="full" 的层计数：5.2 = 21 regular + MTP = 22 层 → 0.21B（shared 层 indexer=None，零参数）
```
MTP 层 indexer 计入（`index_share_for_mtp_iteration=true`，layer 78 有权重）。

**(c3) 混层**：`mlp_layer_types` 数组：dense 层 `3×H×denseI`，sparse 层 `E×3×H×moeI + shared 3×H×moeI×n_shared + router E×H + sharedGate H`。5.2 = 3 dense + 75 sparse + MTP(sparse)；4.7 = 3 dense + 89 sparse + MTP(sparse)。

**(c4) MTP 两族**（事实）：
- glm5 族（5.2）：完整层（MLA+indexer+MoE）+ `eh_proj[2H→H]` + `shared_head.norm` + 各 norm。**无独立 embed/head**。
- glm4 族（4.7）：完整层（GQA+MoE）+ eh_proj + **独立 `embed_tokens[V,H]` + `shared_head.head[V,H]`**（+2×V×H≈1.55B）+ norms。
- 官方训练口径（GLM-5 报告）：训练时 3 个 MTP 层共享参数（参数显存按 1 层计）；工具的 mtp=1 与 checkpoint 一致，注释里标注训练侧 ×3 激活即可（推断标注）。

**(c5) 顶层**：`tie_word_embeddings=false`（embed+lm_head 各一份，5.2 vocab=154880；4.7 vocab=151552）；`final norm`；5.2 无 QKV bias（4.7 有，14336/层）。

### 3.4 任务 D（P0/P1）：`compute` 的 GLM 口径

- **TP 切分**（事实，官方 `base_model_tp_plan`）：`q_b/kv_b` colwise、`o` rowwise、experts `moe_tp_experts`、dense/shared MLP 常规切；**`q_a`、`kv_a`、indexer、router、embed/lm_head/norm 为 TP 复制项**。实现：`nDense` 拆成「可切 / 复制」两桶，`N_rank = 可切/(TP×PP) + 复制×1 + N_expert/(EP×ETP×PP)`。EP：experts `grouped_gemm`，ETP 沿用。
- **优化器 = Muon**（事实，GLM-4.5/5 报告）：2D 权重 fp32 momentum ≈ **4 B/参数**；embed/bias/RMSNorm 用 AdamW（12B）。UI 加 `optimizer ∈ {Adam(×12), Muon(×4)+AdamW(小项)}`，GLM 默认 Muon。保留 `Adam 在 CPU` offload 开关语义（优化器项→0）。
- **CE 峰值**：`2×T×(V/TP)×4` 保留，新增「CE 分 chunk」选项（官方 sequence-chunked output projection，峰值 ÷ chunk 数；默认 1=不分）。
- **激活**：5.2 维持 full recompute 默认 0；非重计算公式的 GLM 修正（indexer scores 只在 full 层、fp32）标 P1，拿到真实 profile 前不承诺精度（推断标注）。
- **MTP 摆放提示**（P2）：官方把 MTP output 与主 output 同 stage、embedding+transformer 在倒数第二 stage——PP>1 时对最后两级 stage 的权重分布加提示文本（事实，报告 §2.4）。

### 3.5 任务 E：验收与测试

1. **参数量单测**（硬门槛）：对 §1.3 的 4 个真值断言（5.2/5.1/4.7/4.7-Flash）。可在浏览器 console 直接调 `parseConfig+countParams`（页面函数是全局的），或抽 Node 快测（页面无依赖，`<script>` 段可整段拷出）。
2. **证据核对法**：ModelScope `config.json` + `model.safetensors.index.json` 的 `total_size`/分片字节数（注意 4.x 的 total_size 是参数量不是字节，v0.4 §10）。
3. **显存 sanity**：GLM-5.2 bf16 全量 = 1506.66GB；TP=8/EP=32 下单卡权重下界 ≈ 1506.66/… 量级检查；与 vLLM 部署事实互证（FP8 8×H200 单节点可服务 → FP8 权重 755.62GB + KV + runtime ≤ 8×141GB）。
4. 页面交互回归：dense 模型隐藏 EP/ETP、`syncSplitUi`、本地预设不受影响。

### 3.6 明确不做（本期）

- 5.3-Flash（KDA/MHC/kpool/ViT/FP8 混合）——公式已备好（v0.4 §4.5–4.8）但工作量大，单独排期。
- fsdp.html / infer.html 的 GLM 化（二期）。
- Flight Recorder / 运行时采集（WEIY-8 另一条线）。

---

## 4. 证据与参考索引（实现时按需取用）

| 主题 | 来源 | 关键内容 |
|---|---|---|
| 结构差异全表 + 公式 | 《对比文档 v0.4》（本次附件） | §1 速览、§3 解析层、§4 公式层、§9 待验证状态 |
| GLM-5 技术报告 | arXiv 2602.15763v2 | Muon/训练基础设施/DSA 训练配方/MLA-256/MTP 共享 |
| IndexCache 论文 | arXiv 2603.12201 | indexer 共享语义（F/S 层、T_cache、蒸馏损失） |
| 官方模型实现 | huggingface/transformers `models/glm_moe_dsa/`（+`glm5_next/` 备用） | indexer 共享、tp/ep/fsdp plan、config 加载逻辑（head_dim 覆盖） |
| 官方仓库 | github.com/zai-org/GLM-5 | 模型矩阵、微调框架（ms-swift/Slime）、ascend.md 部署优化 |
| GLM-5.2 config 真值 | ModelScope `ZhipuAI/GLM-5.2/resolve/master/config.json` | §3.2 映射表的权威 key 集 |
| 部署基准 | recipes.vllm.ai `/zai-org/GLM-5.2`、`/zai-org/GLM-4.7` | FP8 单节点配置、MTP 投机 token 数、1M ctx 约束 |
| GLM-4.5 报告（Muon 细节） | arXiv 2508.06471 | Muon NS=5/μ=0.95/RMS 0.2、embed/bias/norm 用 AdamW |

**仍未解决的输入**（实现时不要假装知道，按「推断」标注）：官方训练并行拓扑数字（TP/PP/EP/DP 具体取值）、真实训练显存 profile、非重计算激活精确值。这些属于 WEIY-8 校准闭环，不是本期阻塞项——参数静态项（权重/梯度/优化器/CE）已可做到 <0.1% 对齐。
