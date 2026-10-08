(function (root, factory) {
  const catalog = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = catalog;
  } else {
    root.MemoryModelCatalog = catalog;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_MODEL_ID = "qwen3.5-27b";

  const SERIES_DEFINITIONS = [
    { seriesId: "qwen-3.8", seriesLabel: "Qwen3.8", defaultModelId: "qwen3.8-27b" },
    { seriesId: "qwen-3.6", seriesLabel: "Qwen3.6", defaultModelId: "qwen3.6-27b" },
    { seriesId: "qwen-3.5", seriesLabel: "Qwen3.5", defaultModelId: DEFAULT_MODEL_ID },
    { seriesId: "qwen-3-vl", seriesLabel: "Qwen3 VL", defaultModelId: "qwen3-vl-8b-instruct" },
    { seriesId: "qwen-3", seriesLabel: "Qwen3", defaultModelId: "qwen3-8b" },
    { seriesId: "qwen-2.5-vl", seriesLabel: "Qwen2.5 VL", defaultModelId: "qwen2.5-vl-7b-instruct" },
    { seriesId: "glm-5", seriesLabel: "GLM-5.x", defaultModelId: "glm-5.2" },
  ].map(Object.freeze);

  const SERIES_BY_ID = Object.create(null);
  for (const series of SERIES_DEFINITIONS) SERIES_BY_ID[series.seriesId] = series;

  const QWEN_HYBRID = Object.freeze({
    vendor: "qwen",
    architecture: "qwen-hybrid-attention",
    moe: false,
    vision: true,
    mtp: true,
    quantization: null,
    independentArchitecture: false,
  });
  const QWEN_TEXT = Object.freeze({
    vendor: "qwen",
    architecture: "qwen3",
    moe: false,
    vision: false,
    mtp: false,
    quantization: null,
    independentArchitecture: false,
  });
  const QWEN_VL = Object.freeze({
    vendor: "qwen",
    architecture: "qwen-vl",
    moe: false,
    vision: true,
    mtp: false,
    quantization: null,
    independentArchitecture: false,
  });
  const GLM_DSA = Object.freeze({
    vendor: "zhipuai",
    architecture: "glm-moe-dsa",
    moe: true,
    vision: false,
    mla: true,
    dsa: true,
    mtp: true,
    muon: true,
    quantization: null,
    independentArchitecture: false,
  });

  function withCapabilities(base, overrides) {
    return Object.freeze(Object.assign({}, base, overrides || {}));
  }

  function defineModel(seriesId, modelId, label, repo, options) {
    const series = SERIES_BY_ID[seriesId];
    if (!series) throw new Error("Unknown model series: " + seriesId);

    const opts = options || {};
    return Object.freeze({
      seriesId,
      seriesLabel: series.seriesLabel,
      modelId,
      label,
      repo,
      source: opts.source || "remote",
      supported: opts.supported !== false,
      capabilities: opts.capabilities || QWEN_TEXT,
      supportNote: opts.supportNote || "",
    });
  }

  function qwen(seriesId, modelId, label, options) {
    const opts = Object.assign({}, options || {});
    const base = opts.capabilities || QWEN_TEXT;
    opts.capabilities = withCapabilities(base, {
      moe: /-A\d+B(?:-|$|\s|\))/.test(label),
    });
    return defineModel(seriesId, modelId, label, "Qwen/" + label.replace(/ \(\d+L\)$/, ""), opts);
  }

  const MODELS = [
    qwen("qwen-3.8", "qwen3.8-27b", "Qwen3.8-27B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.8", "qwen3.8-2.4t-a95b", "Qwen3.8-2.4T-A95B", { capabilities: QWEN_HYBRID }),

    qwen("qwen-3.6", "qwen3.6-27b", "Qwen3.6-27B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.6", "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.6", "qwen3.6-35b-a3b-20l", "Qwen3.6-35B-A3B (20L)", {
      capabilities: QWEN_HYBRID,
      source: "local:20L",
    }),

    qwen("qwen-3.5", "qwen3.5-35b-a3b-12l", "Qwen3.5-35B-A3B (12L)", {
      capabilities: QWEN_HYBRID,
      source: "local:12L",
    }),
    qwen("qwen-3.5", "qwen3.5-0.8b", "Qwen3.5-0.8B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-2b", "Qwen3.5-2B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-4b", "Qwen3.5-4B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-9b", "Qwen3.5-9B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-27b", "Qwen3.5-27B", {
      capabilities: QWEN_HYBRID,
      source: "local:27B",
    }),
    qwen("qwen-3.5", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-122b-a10b", "Qwen3.5-122B-A10B", { capabilities: QWEN_HYBRID }),
    qwen("qwen-3.5", "qwen3.5-397b-a17b", "Qwen3.5-397B-A17B", { capabilities: QWEN_HYBRID }),

    qwen("qwen-3-vl", "qwen3-vl-8b-instruct", "Qwen3-VL-8B-Instruct", { capabilities: QWEN_VL }),
    qwen("qwen-3-vl", "qwen3-vl-32b-instruct", "Qwen3-VL-32B-Instruct", { capabilities: QWEN_VL }),
    qwen("qwen-3-vl", "qwen3-vl-30b-a3b-instruct", "Qwen3-VL-30B-A3B-Instruct", { capabilities: QWEN_VL }),
    qwen("qwen-3-vl", "qwen3-vl-235b-a22b-instruct", "Qwen3-VL-235B-A22B-Instruct", { capabilities: QWEN_VL }),

    qwen("qwen-3", "qwen3-0.6b", "Qwen3-0.6B"),
    qwen("qwen-3", "qwen3-1.7b", "Qwen3-1.7B"),
    qwen("qwen-3", "qwen3-4b", "Qwen3-4B"),
    qwen("qwen-3", "qwen3-8b", "Qwen3-8B"),
    qwen("qwen-3", "qwen3-14b", "Qwen3-14B"),
    qwen("qwen-3", "qwen3-32b", "Qwen3-32B"),
    qwen("qwen-3", "qwen3-4b-instruct", "Qwen3-4B-Instruct"),
    qwen("qwen-3", "qwen3-8b-instruct", "Qwen3-8B-Instruct"),
    qwen("qwen-3", "qwen3-14b-instruct", "Qwen3-14B-Instruct"),
    qwen("qwen-3", "qwen3-32b-instruct", "Qwen3-32B-Instruct"),
    qwen("qwen-3", "qwen3-30b-a3b", "Qwen3-30B-A3B"),
    qwen("qwen-3", "qwen3-30b-a3b-thinking-2507", "Qwen3-30B-A3B-Thinking-2507"),
    qwen("qwen-3", "qwen3-235b-a22b", "Qwen3-235B-A22B"),
    qwen("qwen-3", "qwen3-235b-a22b-instruct-2507", "Qwen3-235B-A22B-Instruct-2507"),
    qwen("qwen-3", "qwen3-next-80b-a3b-instruct", "Qwen3-Next-80B-A3B-Instruct"),
    qwen("qwen-3", "qwen3-coder-30b-a3b-instruct", "Qwen3-Coder-30B-A3B-Instruct"),
    qwen("qwen-3", "qwen3-coder-480b-a35b-instruct", "Qwen3-Coder-480B-A35B-Instruct"),

    qwen("qwen-2.5-vl", "qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct", { capabilities: QWEN_VL }),
    qwen("qwen-2.5-vl", "qwen2.5-vl-72b-instruct", "Qwen2.5-VL-72B-Instruct", { capabilities: QWEN_VL }),

    defineModel("glm-5", "glm-5.2", "GLM-5.2", "ZhipuAI/GLM-5.2", {
      capabilities: GLM_DSA,
    }),
    defineModel("glm-5", "glm-5.3", "GLM-5.3", "ZhipuAI/GLM-5.3", {
      capabilities: withCapabilities(GLM_DSA, { quantization: "fp8-mixed" }),
    }),
    defineModel("glm-5", "glm-5.3-bf16", "GLM-5.3-BF16", "ZhipuAI/GLM-5.3-BF16", {
      capabilities: withCapabilities(GLM_DSA, { quantization: "bf16" }),
    }),
    defineModel("glm-5", "glm-5.2-8l", "GLM-5.2 (8L)", "ZhipuAI/GLM-5.2", {
      capabilities: GLM_DSA,
      source: "glm-scale:8",
      supportNote: "合成减层配置：按 GLM-5.2 结构规则缩放，无 checkpoint 真值，用于小卡数验证",
    }),
    defineModel("glm-5", "glm-5.2-16l", "GLM-5.2 (16L)", "ZhipuAI/GLM-5.2", {
      capabilities: GLM_DSA,
      source: "glm-scale:16",
      supportNote: "合成减层配置：按 GLM-5.2 结构规则缩放，无 checkpoint 真值，用于小卡数验证",
    }),
    defineModel("glm-5", "glm-5.3-flash", "GLM-5.3-Flash", "ZhipuAI/GLM-5.3-Flash", {
      supported: false,
      capabilities: Object.freeze({
        vendor: "zhipuai",
        architecture: "glm5-next",
        moe: true,
        vision: true,
        mla: true,
        dsa: true,
        kda: true,
        mhc: true,
        mtp: true,
        muon: true,
        quantization: "fp8-mixed",
        independentArchitecture: true,
      }),
      supportNote: "独立 KDA/MHC/ViT/FP8 混合架构，当前公式不支持",
    }),
  ];

  const MODEL_BY_ID = Object.create(null);
  const MODEL_BY_LABEL = Object.create(null);
  for (const model of MODELS) {
    MODEL_BY_ID[model.modelId] = model;
    MODEL_BY_LABEL[model.label] = model;
  }

  function listSeries() {
    return SERIES_DEFINITIONS.slice();
  }

  function listModels(seriesId) {
    if (seriesId == null) return MODELS.slice();
    if (!SERIES_BY_ID[seriesId]) return [];
    return MODELS.filter(function (model) { return model.seriesId === seriesId; });
  }

  function findModel(modelIdOrLabel) {
    return MODEL_BY_ID[modelIdOrLabel] || MODEL_BY_LABEL[modelIdOrLabel] || null;
  }

  function findSeriesForModel(modelIdOrLabel) {
    const model = findModel(modelIdOrLabel);
    return model ? SERIES_BY_ID[model.seriesId] : null;
  }

  return Object.freeze({
    listSeries,
    listModels,
    findModel,
    findSeriesForModel,
    defaultModelId: DEFAULT_MODEL_ID,
  });
});
