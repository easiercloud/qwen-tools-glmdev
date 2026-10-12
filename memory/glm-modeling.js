(function (root, factory) {
  const policies = typeof module === "object" && module.exports
    ? require("./training-policies.js")
    : root.MemoryTrainingPolicies;
  const api = factory(policies);
  if (typeof module === "object" && module.exports) module.exports = api;
  root.GlmMemoryModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (TrainingPolicies) {
  "use strict";

  if (!TrainingPolicies) throw new Error("glm-modeling.js 需要先加载 training-policies.js");

  const GIB = 1024 ** 3;
  const GLM_MOE_DSA = "glm_moe_dsa";

  const PRESET_META = {
    "glm-5.2": {
      modelId: "glm-5.2",
    },
    "glm-5.3": {
      modelId: "glm-5.3",
    },
  };

  function makeIndexerTypes(layers, freq, offset) {
    const safeFreq = Math.max(Number(freq) || 1, 1);
    const safeOffset = Number.isFinite(Number(offset)) ? Number(offset) : 2;
    return Array.from({ length: layers }, (_, i) =>
      Math.max(i - safeOffset + 1, 0) % safeFreq === 0 ? "full" : "shared"
    );
  }

  function makeMlpLayerTypes(layers, firstDense) {
    const dense = Math.min(Math.max(Number(firstDense) || 0, 0), layers);
    return Array.from({ length: layers }, (_, i) => i < dense ? "dense" : "sparse");
  }

  function makePresetConfig(modelId, layerOverride) {
    const meta = PRESET_META[modelId];
    if (!meta) return null;
    const override = Number(layerOverride);
    const layers = Number.isFinite(override) && override > 0
      ? Math.floor(override)
      : (meta.layers || 78);
    return {
      architectures: ["GlmMoeDsaForCausalLM"],
      model_type: GLM_MOE_DSA,
      dtype: "bfloat16",
      vocab_size: 154880,
      hidden_size: 6144,
      num_hidden_layers: layers,
      num_attention_heads: 64,
      num_key_value_heads: 64,
      head_dim: 192,
      q_lora_rank: 2048,
      kv_lora_rank: 512,
      qk_nope_head_dim: 192,
      qk_rope_head_dim: 64,
      v_head_dim: 256,
      attention_bias: false,
      intermediate_size: 12288,
      moe_intermediate_size: 2048,
      n_routed_experts: 256,
      n_shared_experts: 1,
      num_experts_per_tok: 8,
      first_k_dense_replace: 3,
      mlp_layer_types: makeMlpLayerTypes(layers, 3),
      index_topk: 2048,
      index_head_dim: 128,
      index_n_heads: 32,
      index_topk_freq: 4,
      index_skip_topk_offset: 3,
      indexer_types: makeIndexerTypes(layers, 4, 3),
      index_share_for_mtp_iteration: true,
      num_nextn_predict_layers: 1,
      tie_word_embeddings: false,
      moe_router_dtype: "float32",
      topk_method: "noaux_tc",
      scoring_func: "sigmoid",
      // Architecture presets intentionally do not encode checkpoint format.
      // BF16/FP8/W4/W8 are selected independently as artifacts.
      quantization_config: null,
    };
  }

  function withLayerOverride(cfg, layers) {
    const n = Number(layers);
    if (!cfg || typeof cfg !== "object" || !Number.isFinite(n) || Math.floor(n) !== n || n < 1) {
      throw new Error("withLayerOverride 需要正整数层数");
    }
    const clone = JSON.parse(JSON.stringify(cfg));
    const text = clone.text_config || clone.language_config || clone;
    text.num_hidden_layers = Math.floor(n);
    delete text.mlp_layer_types;
    delete text.indexer_types;
    delete clone.mlp_layer_types;
    delete clone.indexer_types;
    return clone;
  }

  function textConfigOf(cfg) {
    return cfg && (cfg.text_config || cfg.language_config || cfg);
  }

  function isGlmMoeDsaConfig(cfg) {
    const text = textConfigOf(cfg) || {};
    return text.model_type === GLM_MOE_DSA
      || (Array.isArray(cfg && cfg.architectures)
        && cfg.architectures.includes("GlmMoeDsaForCausalLM"));
  }

  function requiredPositive(text, key) {
    const value = Number(text[key]);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error("GLM config 缺少有效字段 " + key);
    }
    return value;
  }

  function parseConfig(cfg, metadata) {
    if (!isGlmMoeDsaConfig(cfg)) {
      const text = textConfigOf(cfg) || {};
      if (text.model_type === "glm5_next") {
        throw new Error("GLM-5.3-Flash 需要 KDA/MHC/ViT 专用公式，当前未启用");
      }
      throw new Error("不是受支持的 GLM MoE-DSA 配置");
    }
    const text = textConfigOf(cfg);
    const meta = metadata || {};
    const layers = requiredPositive(text, "num_hidden_layers");
    const firstDense = Number.isFinite(Number(text.first_k_dense_replace))
      ? Number(text.first_k_dense_replace)
      : 3;
    const mlpLayerTypes = Array.isArray(text.mlp_layer_types)
      ? text.mlp_layer_types.slice()
      : makeMlpLayerTypes(layers, firstDense);
    const indexerTypes = Array.isArray(text.indexer_types)
      ? text.indexer_types.slice()
      : makeIndexerTypes(layers, text.index_topk_freq, text.index_skip_topk_offset);
    if (mlpLayerTypes.length !== layers) {
      throw new Error("GLM mlp_layer_types 长度与 num_hidden_layers 不一致");
    }
    if (indexerTypes.length !== layers) {
      throw new Error("GLM indexer_types 长度与 num_hidden_layers 不一致");
    }
    if (mlpLayerTypes.some((x) => x !== "dense" && x !== "sparse")) {
      throw new Error("GLM mlp_layer_types 含未知类型");
    }
    if (indexerTypes.some((x) => x !== "full" && x !== "shared")) {
      throw new Error("GLM indexer_types 含未知类型");
    }

    const modelId = meta.modelId || "";
    const presetMeta = PRESET_META[modelId] || {};
    const overrideRaw = Number(meta.layerOverride);
    const layerOverride = Number.isFinite(overrideRaw) && overrideRaw > 0 ? Math.floor(overrideRaw) : null;
    const scaled = layerOverride != null || !!presetMeta.scaled;
    const quant = text.quantization_config || cfg.quantization_config || null;
    const artifact = TrainingPolicies.resolveArtifact({
      modelId,
      artifactId: meta.artifactId,
      rawAlias: meta.rawAlias,
    });
    return {
      family: "glm",
      architectureKind: GLM_MOE_DSA,
      arch: (cfg.architectures || [text.model_type || ""]).join(", "),
      modelId,
      vocab: requiredPositive(text, "vocab_size"),
      hidden: requiredPositive(text, "hidden_size"),
      layers,
      nQ: requiredPositive(text, "num_attention_heads"),
      nKV: requiredPositive(text, "num_key_value_heads"),
      qLoraRank: requiredPositive(text, "q_lora_rank"),
      kvLoraRank: requiredPositive(text, "kv_lora_rank"),
      qkNopeHeadDim: requiredPositive(text, "qk_nope_head_dim"),
      qkRopeHeadDim: requiredPositive(text, "qk_rope_head_dim"),
      vHeadDim: requiredPositive(text, "v_head_dim"),
      experts: requiredPositive(text, "n_routed_experts"),
      nSharedExperts: requiredPositive(text, "n_shared_experts"),
      moeI: requiredPositive(text, "moe_intermediate_size"),
      denseI: requiredPositive(text, "intermediate_size"),
      shareI: requiredPositive(text, "n_shared_experts") * requiredPositive(text, "moe_intermediate_size"),
      topk: requiredPositive(text, "num_experts_per_tok"),
      mlpLayerTypes,
      indexerTypes,
      indexTopk: requiredPositive(text, "index_topk"),
      indexHeadDim: requiredPositive(text, "index_head_dim"),
      indexNHeads: requiredPositive(text, "index_n_heads"),
      indexShareForMtp: text.index_share_for_mtp_iteration !== false,
      mtp: Math.max(Number(text.num_nextn_predict_layers) || 0, 0),
      tie: !!(cfg.tie_word_embeddings ?? text.tie_word_embeddings),
      attentionBias: !!text.attention_bias,
      routerFp32: text.moe_router_dtype === "float32" || text.moe_router_dtype === "fp32",
      quantization: quant,
      artifact,
      artifactId: artifact.artifactId,
      checkpointPrecision: artifact.label,
      checkpointTruthBytes: !scaled ? artifact.checkpointBytes : null,
      synthetic: scaled,
      sourceRevision: scaled
        ? "synthetic scaled config（减层合成，非官方 checkpoint）"
        : (meta.sourceRevision || "ModelScope master snapshot 2026-09-28"),
      layerTypes: [],
      headDim: 0,
      fullInterval: 0,
      attnGate: false,
      gdnKH: 0,
      gdnKD: 0,
      gdnVH: 0,
      gdnVD: 0,
      gdnConv: 0,
      vitDepth: 0,
      vitH: 0,
      vitI: 0,
      vitOut: 0,
      patch: 0,
      tPatch: 0,
      inCh: 0,
      merge: 0,
      nPos: 0,
    };
  }

  function withArtifact(model, metadata) {
    if (!model || model.family !== "glm") throw new Error("withArtifact 仅支持 GLM 模型");
    const meta = metadata || {};
    const artifact = TrainingPolicies.resolveArtifact({
      modelId: model.modelId,
      artifactId: meta.artifactId,
      rawAlias: meta.rawAlias,
    });
    return Object.assign({}, model, {
      artifact,
      artifactId: artifact.artifactId,
      checkpointPrecision: artifact.label,
      checkpointTruthBytes: model.synthetic ? null : artifact.checkpointBytes,
    });
  }

  function buildComponents(model) {
    if (!model || model.architectureKind !== GLM_MOE_DSA) {
      throw new Error("buildComponents 仅支持 glm_moe_dsa");
    }
    const components = [];
    const add = (key, label, unitNumel, repeat, options) => {
      const opts = options || {};
      const safeRepeat = repeat == null ? 1 : repeat;
      const numel = unitNumel * safeRepeat;
      if (!Number.isFinite(numel) || numel < 0) throw new Error("非法组件参数量: " + key);
      if (numel === 0) return;
      components.push({
        key,
        label,
        unitNumel,
        repeat: safeRepeat,
        numel,
        parallel: opts.parallel || "replicated",
        optimizer: opts.optimizer || "muon",
        dtypeBytes: opts.dtypeBytes || 2,
        trainable: opts.trainable !== false,
        group: opts.group || "other",
        mtpNumel: unitNumel * (opts.mtpRepeat || 0),
      });
    };

    const V = model.vocab;
    const H = model.hidden;
    const L = model.layers;
    const mtp = model.mtp || 0;
    const totalDecoderLayers = L + mtp;
    const sparseRegular = model.mlpLayerTypes.filter((x) => x === "sparse").length;
    const denseRegular = model.mlpLayerTypes.filter((x) => x === "dense").length;
    const sparseTotal = sparseRegular + mtp;
    const fullIndexerRegular = model.indexerTypes.filter((x) => x === "full").length;
    const fullIndexerTotal = fullIndexerRegular + mtp;

    add("embed", "词嵌入", V * H, 1, {
      parallel: "replicated", optimizer: "adamw", group: "vocab",
    });
    if (!model.tie) {
      add("lm_head", "输出头", V * H, 1, {
        parallel: "tp", optimizer: "muon", group: "vocab",
      });
    }
    add("final_norm", "最终 RMSNorm", H, 1, {
      parallel: "replicated", optimizer: "adamw", group: "norm",
    });

    add("mla_q_a", "MLA q_a_proj", H * model.qLoraRank, totalDecoderLayers, {
      parallel: "replicated", group: "mla", mtpRepeat: mtp,
    });
    add("mla_q_a_norm", "MLA q_a_layernorm", model.qLoraRank, totalDecoderLayers, {
      parallel: "replicated", optimizer: "adamw", group: "norm", mtpRepeat: mtp,
    });
    add("mla_q_b", "MLA q_b_proj", model.qLoraRank * model.nQ * (model.qkNopeHeadDim + model.qkRopeHeadDim), totalDecoderLayers, {
      parallel: "tp", group: "mla", mtpRepeat: mtp,
    });
    add("mla_kv_a", "MLA kv_a_proj_with_mqa", H * (model.kvLoraRank + model.qkRopeHeadDim), totalDecoderLayers, {
      parallel: "replicated", group: "mla", mtpRepeat: mtp,
    });
    add("mla_kv_a_norm", "MLA kv_a_layernorm", model.kvLoraRank, totalDecoderLayers, {
      parallel: "replicated", optimizer: "adamw", group: "norm", mtpRepeat: mtp,
    });
    add("mla_kv_b", "MLA kv_b_proj", model.kvLoraRank * model.nQ * (model.qkNopeHeadDim + model.vHeadDim), totalDecoderLayers, {
      parallel: "tp", group: "mla", mtpRepeat: mtp,
    });
    add("mla_o", "MLA o_proj", model.nQ * model.vHeadDim * H, totalDecoderLayers, {
      parallel: "tp", group: "mla", mtpRepeat: mtp,
    });
    if (model.attentionBias) {
      add("mla_q_a_bias", "MLA q_a bias", model.qLoraRank, totalDecoderLayers, {
        parallel: "replicated", optimizer: "adamw", group: "bias", mtpRepeat: mtp,
      });
      add("mla_kv_a_bias", "MLA kv_a bias", model.kvLoraRank + model.qkRopeHeadDim, totalDecoderLayers, {
        parallel: "replicated", optimizer: "adamw", group: "bias", mtpRepeat: mtp,
      });
      add("mla_o_bias", "MLA o bias", H, totalDecoderLayers, {
        parallel: "replicated", optimizer: "adamw", group: "bias", mtpRepeat: mtp,
      });
    }

    add("indexer_wq_b", "DSA indexer wq_b", model.qLoraRank * model.indexNHeads * model.indexHeadDim, fullIndexerTotal, {
      parallel: "replicated", group: "indexer", mtpRepeat: mtp,
    });
    add("indexer_wk", "DSA indexer wk", H * model.indexHeadDim, fullIndexerTotal, {
      parallel: "replicated", group: "indexer", mtpRepeat: mtp,
    });
    add("indexer_weights", "DSA indexer weights_proj", H * model.indexNHeads, fullIndexerTotal, {
      parallel: "replicated", group: "indexer", mtpRepeat: mtp,
    });
    add("indexer_norm", "DSA indexer LayerNorm", 2 * model.indexHeadDim, fullIndexerTotal, {
      parallel: "replicated", optimizer: "adamw", group: "norm", mtpRepeat: mtp,
    });

    add("dense_mlp", "Dense MLP", 3 * H * model.denseI, denseRegular, {
      parallel: "tp", group: "dense_mlp",
    });
    add("routed_experts", "路由专家", model.experts * 3 * H * model.moeI, sparseTotal, {
      parallel: "expert", group: "experts", mtpRepeat: mtp,
    });
    add("router", "MoE router", model.experts * H, sparseTotal, {
      parallel: "replicated", group: "router", mtpRepeat: mtp,
    });
    add("shared_experts", "共享专家", 3 * H * model.shareI, sparseTotal, {
      parallel: "tp", group: "shared_experts", mtpRepeat: mtp,
    });
    add("router_correction", "router correction buffer", model.experts, sparseTotal, {
      parallel: "replicated", optimizer: "none", dtypeBytes: 4, trainable: false, group: "buffer", mtpRepeat: mtp,
    });
    add("decoder_norms", "Decoder RMSNorm", 2 * H, totalDecoderLayers, {
      parallel: "replicated", optimizer: "adamw", group: "norm", mtpRepeat: mtp,
    });

    add("mtp_eh_proj", "MTP eh_proj", 2 * H * H, mtp, {
      parallel: "tp", group: "mtp", mtpRepeat: mtp,
    });
    add("mtp_aux_norms", "MTP enorm/hnorm/shared_head.norm", 3 * H, mtp, {
      parallel: "replicated", optimizer: "adamw", group: "norm", mtpRepeat: mtp,
    });

    return components;
  }

  function summarizeComponents(model, components) {
    const list = components || buildComponents(model);
    const sum = (predicate, map) => list
      .filter(predicate)
      .reduce((total, item) => total + (map ? map(item) : item.numel), 0);
    const N = sum(() => true);
    const trainableN = sum((x) => x.trainable);
    const bufferN = sum((x) => !x.trainable);
    const checkpointBytes = sum(() => true, (x) => x.numel * x.dtypeBytes);
    const sparseRegular = model.mlpLayerTypes.filter((x) => x === "sparse").length;
    const denseRegular = model.mlpLayerTypes.filter((x) => x === "dense").length;
    const fullIndexerRegular = model.indexerTypes.filter((x) => x === "full").length;
    const groupN = (group) => sum((x) => x.group === group);
    return {
      components: list,
      N,
      trainableN,
      bufferN,
      checkpointBytes,
      checkpointTruthBytes: model.checkpointTruthBytes,
      checkpointDeltaBytes: model.checkpointTruthBytes == null ? null : checkpointBytes - model.checkpointTruthBytes,
      nExpert: sum((x) => x.parallel === "expert"),
      nTpShard: sum((x) => x.parallel === "tp"),
      nTpReplicated: sum((x) => x.parallel === "replicated"),
      nMuon: sum((x) => x.trainable && x.optimizer === "muon"),
      nAdamw: sum((x) => x.trainable && x.optimizer === "adamw"),
      denseRegular,
      sparseRegular,
      sparseTotal: sparseRegular + (model.mtp || 0),
      fullIndexerRegular,
      fullIndexerTotal: fullIndexerRegular + (model.mtp || 0),
      embed: groupN("vocab") ? list.find((x) => x.key === "embed").numel : 0,
      lm: list.find((x) => x.key === "lm_head")?.numel || 0,
      finalNorm: list.find((x) => x.key === "final_norm")?.numel || 0,
      mlaN: groupN("mla"),
      indexerN: groupN("indexer"),
      denseMlpN: groupN("dense_mlp"),
      expertN: groupN("experts"),
      routerN: groupN("router"),
      sharedN: groupN("shared_experts"),
      normN: groupN("norm"),
      mtpExtraN: groupN("mtp"),
      bufferBytes: sum((x) => !x.trainable, (x) => x.numel * x.dtypeBytes),
      V: model.vocab,
      H: model.hidden,
      L: model.layers,
      E: model.experts,
      I: model.moeI,
      topk: model.topk,
      mtp: model.mtp || 0,
      isGlm: true,
    };
  }

  function positive(name, value) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(name + " 必须大于 0");
    return value;
  }

  function rankScale(component, train) {
    const TP = positive("TP", train.TP);
    const PP = positive("PP", train.PP);
    const EP = positive("EP", train.EP);
    const ETP = positive("ETP", train.ETP);
    if (component.parallel === "expert") return 1 / (EP * ETP * PP);
    if (component.parallel === "tp") return 1 / (TP * PP);
    return 1 / PP;
  }

  function aggregateStatic(model, train, components) {
    const list = components || buildComponents(model);
    const optimizerType = train.optimizer || "muon";
    const weightBytes = Number.isFinite(train.bw) && train.bw > 0 ? train.bw : 2;
    const gradBytes = Number.isFinite(train.gradBytes) && train.gradBytes > 0 ? train.gradBytes : 2;
    const memory = TrainingPolicies.aggregateComponentMemory({
      components: list,
      rankScale: function (component) { return rankScale(component, train); },
      mode: train.trainingMode || train.mode || "pretrain",
      artifact: model.artifact || TrainingPolicies.resolveArtifact({
        modelId: model.modelId,
        artifactId: model.artifactId,
      }),
      trainableRatio: train.trainableRatio,
      trainableParams: train.trainableParams,
      trainWeightBytes: weightBytes,
      gradBytes,
      mtpEnabled: train.mtpEnabled !== false,
      optimizerType,
      optimizerOffload: !!train.optimizerOffload || !!train.adamOff,
    });
    return {
      perRank: memory.perRank,
      rankBase: memory.rankBase,
      rankTrainable: memory.rankTrainable,
      rankBaseTrainable: memory.rankBaseTrainable,
      rankAdapter: memory.rankAdapter,
      rankFrozen: memory.rankFrozen,
      rankBuffer: memory.rankBuffer,
      rankMuon: memory.rankMuon,
      rankAdamw: memory.rankAdamw,
      W: memory.weightBytes / GIB,
      G: memory.gradBytes / GIB,
      optimizer: memory.optimizerBytes / GIB,
      optimizerType,
      policy: memory.policy,
      nRankExpert: memory.nRankExpert,
      nRankTpShard: memory.nRankTpShard,
      nRankReplicated: memory.nRankReplicated,
    };
  }

  function computeDynamic(model, train) {
    const TP = positive("TP", train.TP);
    const chunks = positive("CE chunks", train.ceChunks == null ? 1 : train.ceChunks);
    const mbs = Number(train.mbs);
    const prompt = Number(train.prompt);
    const response = Number(train.resp);
    if (![mbs, prompt, response].every(Number.isFinite) || mbs < 0 || prompt < 0 || response < 0) {
      return { T: null, Vloc: model.vocab / TP, ce: null, dsaIndices: null, routerLogits: null, outputPhase: null, attentionPhase: null, total: null };
    }
    const T = mbs * (prompt + response);
    const Vloc = model.vocab / TP;
    const ce = 2 * T * Vloc * 4 / chunks / GIB;
    const dsaIndices = T * model.indexTopk * 4 / GIB;
    const routerLogits = model.routerFp32 ? T * model.experts * 4 / GIB : 0;
    const outputPhase = ce;
    const attentionPhase = Math.max(dsaIndices, routerLogits);
    return { T, Vloc, ce, dsaIndices, routerLogits, outputPhase, attentionPhase, total: Math.max(outputPhase, attentionPhase) };
  }

  return {
    GLM_MOE_DSA,
    PRESET_META,
    makeIndexerTypes,
    makeMlpLayerTypes,
    makePresetConfig,
    withLayerOverride,
    withArtifact,
    isGlmMoeDsaConfig,
    parseConfig,
    buildComponents,
    summarizeComponents,
    aggregateStatic,
    computeDynamic,
  };
});
