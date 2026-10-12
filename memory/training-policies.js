(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.MemoryTrainingPolicies = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_TRAINABLE_RATIO = 0.001;

  const TRAINING_MODES = [
    {
      modeId: "pretrain",
      label: "预训练 / 继续预训练",
      fullUpdate: true,
      adapterBased: false,
      description: "全部模型参数参与反向与优化器更新。",
    },
    {
      modeId: "full_sft",
      label: "全参 SFT",
      fullUpdate: true,
      adapterBased: false,
      description: "全部模型参数参与监督微调。",
    },
    {
      modeId: "lora",
      label: "LoRA",
      fullUpdate: false,
      adapterBased: true,
      description: "冻结基座，只为 adapter 分配权重、梯度与优化器状态。",
    },
    {
      modeId: "qlora",
      label: "QLoRA",
      fullUpdate: false,
      adapterBased: true,
      requiresQuantizedArtifact: true,
      description: "量化冻结基座，只训练高精度 adapter。",
    },
    {
      modeId: "rl_actor",
      label: "强化学习（actor update）",
      fullUpdate: true,
      adapterBased: false,
      description: "只建模当前 actor 的全参更新；不包含 reference、reward、critic。",
    },
  ].map(Object.freeze);

  const MODE_BY_ID = Object.create(null);
  for (const mode of TRAINING_MODES) MODE_BY_ID[mode.modeId] = mode;

  function alias(raw, hardware, revision, flags) {
    return Object.freeze(Object.assign({
      raw,
      hardware: hardware || null,
      revision: revision || null,
    }, flags || {}));
  }

  const ARTIFACTS = [
    {
      artifactId: "bf16",
      label: "BF16",
      trainingCompatibility: "native",
      weightBits: 16,
      activationBits: 16,
      cacheBits: null,
      format: "bf16",
      quantized: false,
      storageBytesPerParam: 2,
      checkpointBytesByModel: {
        "glm-5.2": 1506659919872,
        "glm-5.3": 1506659919872,
      },
      aliases: [
        alias("GLM-5.2", null, null),
        alias("GLM-5.2-0610-Provider", null, "0610", { provider: true }),
        alias("GLM-5.3-BF16", null, null),
      ],
    },
    {
      artifactId: "fp8",
      label: "FP8",
      trainingCompatibility: "runtime-dependent",
      weightBits: 8,
      activationBits: 8,
      cacheBits: null,
      format: "fp8",
      quantized: true,
      storageBytesPerParam: 1.015625,
      checkpointBytesByModel: {
        "glm-5.2": 755617140416,
        "glm-5.3": 755617140416,
      },
      aliases: [
        alias("GLM-5.2-FP8", null, null),
        alias("GLM-5.3", null, null),
      ],
    },
    {
      artifactId: "w8a8-c8",
      label: "W8A8（可含 C8）",
      trainingCompatibility: "runtime-dependent",
      weightBits: 8,
      activationBits: 8,
      cacheBits: 8,
      format: "int8",
      quantized: true,
      storageBytesPerParam: 1.015625,
      aliases: [
        alias("GLM-5.2-w8a8", null, null),
        alias("GLM-5.2-W8A8C8-A3-0731", "A3", "0731"),
        alias("GLM-5.1-W8A8C8-MTP", null, null, { mtpTagged: true }),
        alias("GLM-5-W8a8-new", null, null),
      ],
    },
    {
      artifactId: "w4a8-c8",
      label: "W4A8（可含 C8）",
      trainingCompatibility: "runtime-dependent",
      weightBits: 4,
      activationBits: 8,
      cacheBits: 8,
      format: "int4-int8",
      quantized: true,
      storageBytesPerParam: 0.515625,
      aliases: [
        alias("GLM-5.2-W4A8", null, null),
        alias("GLM-5.2-Provide-0610-W4A8", null, "0610", { provider: true }),
        alias("GLM-5.2-W4A8C8-0630", null, "0630"),
        alias("GLM-5.2-W4A8C8-0702-MTP", null, "0702", { mtpTagged: true }),
        alias("GLM-5.2-W4A8C8-A3-0731", "A3", "0731"),
        alias("GLM-5.2-W4A8C8-A5-0731", "A5", "0731"),
        alias("GLM-5-w4a8", null, null),
      ],
    },
    {
      artifactId: "w4a4c8-mxfp4",
      label: "W4A4C8 MXFP4",
      trainingCompatibility: "runtime-dependent",
      weightBits: 4,
      activationBits: 4,
      cacheBits: 8,
      format: "mxfp4",
      quantized: true,
      storageBytesPerParam: 0.53125,
      aliases: [
        alias("GLM-5.2-W4A4C8-mxfp4-0713", null, "0713"),
        alias("GLM-5.2-W4A4C8-mxfp4-A5-0731", "A5", "0731"),
        alias("GLM-5.2-W4A4-mxfp4", null, null),
      ],
    },
    {
      artifactId: "w8a8c8-mxfp8",
      label: "W8A8C8 MXFP8",
      trainingCompatibility: "runtime-dependent",
      weightBits: 8,
      activationBits: 8,
      cacheBits: 8,
      format: "mxfp8",
      quantized: true,
      storageBytesPerParam: 1.03125,
      aliases: [
        alias("GLM-5.2-W8A8C8-mxfp8", null, null),
        alias("GLM-5.2-W8A8C8-mxfp8-A5-0731", "A5", "0731"),
        alias("GLM-5.2-W8A8-mxfp8", null, null),
      ],
    },
  ].map(function (artifact) {
    return Object.freeze(Object.assign({}, artifact, {
      aliases: Object.freeze((artifact.aliases || []).slice()),
      checkpointBytesByModel: Object.freeze(Object.assign({}, artifact.checkpointBytesByModel || {})),
    }));
  });

  const ARTIFACT_BY_ID = Object.create(null);
  for (const artifact of ARTIFACTS) ARTIFACT_BY_ID[artifact.artifactId] = artifact;

  const DEFAULT_ARTIFACT_BY_MODEL = Object.freeze({
    "glm-5.2": "bf16",
    "glm-5.3": "fp8",
  });

  function listTrainingModes() {
    return TRAINING_MODES.slice();
  }

  function findTrainingMode(modeId) {
    return MODE_BY_ID[modeId] || null;
  }

  function listArtifacts(modelId) {
    // Formats are framework-level choices and are also useful for Qwen/FSDP.
    // The alias metadata happens to describe the supplied GLM artifacts only.
    return ARTIFACTS.slice();
  }

  function findArtifact(artifactId) {
    return ARTIFACT_BY_ID[artifactId] || null;
  }

  function defaultArtifactId(modelId) {
    return DEFAULT_ARTIFACT_BY_MODEL[modelId] || "bf16";
  }

  function cleanAlias(raw) {
    return String(raw == null ? "" : raw)
      .replace(/&#x20;|&#32;|&nbsp;/gi, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function inferArtifactId(name) {
    const upper = name.toUpperCase();
    if (/MXFP4|W4A4/.test(upper)) return "w4a4c8-mxfp4";
    if (/MXFP8/.test(upper)) return "w8a8c8-mxfp8";
    if (/W4A8/.test(upper)) return "w4a8-c8";
    if (/W8A8/.test(upper)) return "w8a8-c8";
    if (/FP8/.test(upper)) return "fp8";
    return "bf16";
  }

  function parseArtifactAlias(raw) {
    const rawAlias = cleanAlias(raw);
    if (!rawAlias) return null;
    const upper = rawAlias.toUpperCase();
    const releaseMatch = upper.match(/GLM[- ]?5(?:[.](\d))?/);
    const hardwareMatch = upper.match(/(?:^|[-_ ])(A3|A5)(?:[-_ ]|$)/);
    const revisionMatches = Array.from(upper.matchAll(/(?:^|[-_ ])(\d{4})(?=[-_ ]|$)/g));
    const minor = releaseMatch && releaseMatch[1];
    const modelId = minor ? "glm-5." + minor : "glm-5";
    const dflash = /DFLASH/.test(upper);
    const independentFlash = !dflash && /GLM[- ]?5[.]3\s*-?\s*FLASH/.test(upper);
    return Object.freeze({
      rawAlias,
      modelId,
      artifactId: inferArtifactId(rawAlias),
      hardware: hardwareMatch ? hardwareMatch[1] : (/GPU(?:[-_ ]WEIGHT)?/.test(upper) ? "GPU" : null),
      revision: revisionMatches.length ? revisionMatches[revisionMatches.length - 1][1] : null,
      provider: /PROVIDE|PROVIDER/.test(upper),
      mtpTagged: /(?:^|[-_ ])MTP(?:[-_ ]|$)/.test(upper),
      role: dflash ? "drafter" : "base",
      independentArchitecture: independentFlash,
    });
  }

  function resolveArtifact(options) {
    const opts = typeof options === "string" ? { artifactId: options } : (options || {});
    const parsed = opts.rawAlias ? parseArtifactAlias(opts.rawAlias) : null;
    const artifactId = opts.artifactId || (parsed && parsed.artifactId) || defaultArtifactId(opts.modelId);
    const preset = findArtifact(artifactId);
    if (!preset) throw new Error("未知 checkpoint/artifact: " + artifactId);
    const modelId = opts.modelId || (parsed && parsed.modelId) || "";
    return Object.freeze(Object.assign({}, preset, {
      modelId,
      rawAlias: parsed ? parsed.rawAlias : (opts.rawAlias || null),
      hardware: parsed ? parsed.hardware : (opts.hardware || null),
      revision: parsed ? parsed.revision : (opts.revision || null),
      provider: parsed ? parsed.provider : !!opts.provider,
      mtpTagged: parsed ? parsed.mtpTagged : !!opts.mtpTagged,
      role: parsed ? parsed.role : "base",
      independentArchitecture: parsed ? parsed.independentArchitecture : false,
      checkpointBytes: preset.checkpointBytesByModel[modelId] || null,
    }));
  }

  function finiteNonNegative(value) {
    if (value == null || value === "") return null;
    const n = Number(value);
    return Number.isFinite(n) && n >= 0 ? n : null;
  }

  function computeTrainableParams(totalParams, options) {
    const total = finiteNonNegative(totalParams);
    if (total == null) throw new Error("totalParams 必须是非负数");
    const opts = options || {};
    const exact = finiteNonNegative(opts.trainableParams);
    if (exact != null) return Math.min(total, exact);
    const ratioValue = finiteNonNegative(opts.trainableRatio);
    const ratio = ratioValue == null ? DEFAULT_TRAINABLE_RATIO : Math.min(ratioValue, 1);
    return total * ratio;
  }

  function resolveTrainingPolicy(options) {
    const opts = options || {};
    const mode = findTrainingMode(opts.mode || opts.modeId || "pretrain");
    if (!mode) throw new Error("未知训练模式: " + (opts.mode || opts.modeId));
    const artifact = opts.artifact && opts.artifact.artifactId
      ? opts.artifact
      : resolveArtifact({
        modelId: opts.modelId,
        artifactId: opts.artifactId || (typeof opts.artifact === "string" ? opts.artifact : null),
        rawAlias: opts.rawAlias,
      });
    const total = finiteNonNegative(opts.totalTrainableParams);
    const adapterParams = mode.adapterBased
      ? computeTrainableParams(total == null ? 0 : total, opts)
      : 0;
    const ratio = total && mode.adapterBased ? adapterParams / total : 0;
    const warnings = [];
    if (mode.modeId === "qlora" && !artifact.quantized) {
      warnings.push("QLoRA 选择了非量化基座；请改选 W4/W8/FP8 artifact，当前按其实际存储精度计算");
    }
    if (mode.adapterBased && artifact.quantized) {
      warnings.push(artifact.label + " 仅按冻结基座的格式字节估算；能否直接训练取决于框架、kernel 与实际 checkpoint，当前未验证可加载性");
      if (mode.modeId === "lora") {
        warnings.push("LoRA 选择了量化冻结基座；显存语义等同量化 LoRA/QLoRA");
      }
    }
    if (mode.modeId === "rl_actor") {
      warnings.push("强化学习当前只计 actor update，不包含 reference、reward、critic 与 rollout engine");
    }
    if (artifact.role && artifact.role !== "base") warnings.push("DFlash/drafter 不能作为训练基座");
    if (artifact.independentArchitecture) warnings.push("Flash 是独立架构，不能作为普通 GLM-5.3 artifact 套用");
    return Object.freeze({
      modeId: mode.modeId,
      label: mode.label,
      fullUpdate: mode.fullUpdate,
      adapterBased: mode.adapterBased,
      frozenBase: !mode.fullUpdate,
      artifact,
      artifactAffectsBase: !mode.fullUpdate,
      trainWeightBytes: finiteNonNegative(opts.trainWeightBytes) || 2,
      gradBytes: finiteNonNegative(opts.gradBytes) || 2,
      fullOptimizerBytes: finiteNonNegative(opts.fullOptimizerBytes) || 12,
      adapterOptimizerBytes: finiteNonNegative(opts.adapterOptimizerBytes) || 12,
      mtpEnabled: opts.mtpEnabled !== false,
      totalTrainableParams: total,
      adapterParams,
      trainableRatio: ratio,
      warnings: Object.freeze(warnings),
    });
  }

  function aggregateComponentMemory(options) {
    const opts = options || {};
    const components = Array.isArray(opts.components) ? opts.components : [];
    const scale = typeof opts.rankScale === "function" ? opts.rankScale : function () { return 1; };
    const mtpEnabled = opts.policy && opts.policy.modeId
      ? opts.policy.mtpEnabled !== false
      : opts.mtpEnabled !== false;
    const eligibleGlobal = components.reduce(function (sum, component) {
      if (!component.trainable || component.frozen) return sum;
      const mtpNumel = Number(component.mtpNumel) || 0;
      return sum + component.numel - (mtpEnabled ? 0 : mtpNumel);
    }, 0);
    const policy = opts.policy && opts.policy.modeId
      ? opts.policy
      : resolveTrainingPolicy(Object.assign({}, opts, { totalTrainableParams: eligibleGlobal }));
    const adapterRatio = policy.adapterBased && eligibleGlobal > 0
      ? policy.adapterParams / eligibleGlobal
      : 0;
    let rankBase = 0;
    let rankBuffer = 0;
    let rankBaseTrainable = 0;
    let rankAdapter = 0;
    let rankFrozen = 0;
    let rankMuon = 0;
    let rankAdamw = 0;
    let weightRaw = 0;
    let gradRaw = 0;
    let nRankExpert = 0;
    let nRankTpShard = 0;
    let nRankReplicated = 0;

    for (const component of components) {
      const factor = Number(scale(component));
      if (!Number.isFinite(factor) || factor < 0) throw new Error("rankScale 必须返回非负数");
      const rankNumel = component.numel * factor;
      if (component.parallel === "expert") nRankExpert += rankNumel;
      else if (component.parallel === "tp") nRankTpShard += rankNumel;
      else nRankReplicated += rankNumel;

      if (!component.trainable) {
        rankBuffer += rankNumel;
        weightRaw += rankNumel * (component.dtypeBytes || 2);
        continue;
      }

      rankBase += rankNumel;
      if (component.frozen) {
        const frozenBytes = policy.fullUpdate
          ? policy.trainWeightBytes
          : policy.artifact.storageBytesPerParam;
        weightRaw += rankNumel * frozenBytes;
        rankFrozen += rankNumel;
        continue;
      }
      const mtpRank = (Number(component.mtpNumel) || 0) * factor;
      const eligibleRank = Math.max(rankNumel - (policy.mtpEnabled ? 0 : mtpRank), 0);
      const baseBytes = policy.fullUpdate
        ? policy.trainWeightBytes
        : policy.artifact.storageBytesPerParam;
      weightRaw += rankNumel * baseBytes;

      if (policy.fullUpdate) {
        rankBaseTrainable += eligibleRank;
        rankFrozen += rankNumel - eligibleRank;
        gradRaw += eligibleRank * policy.gradBytes;
        if (component.optimizer === "adamw") rankAdamw += eligibleRank;
        else if (component.optimizer !== "none") rankMuon += eligibleRank;
      } else {
        const adapterRank = eligibleRank * adapterRatio;
        rankFrozen += rankNumel;
        rankAdapter += adapterRank;
        rankAdamw += adapterRank;
        weightRaw += adapterRank * policy.trainWeightBytes;
        gradRaw += adapterRank * policy.gradBytes;
      }
    }

    const rankTrainable = rankBaseTrainable + rankAdapter;
    let optimizerRaw = 0;
    if (!opts.optimizerOffload && !opts.adamOff) {
      if (policy.adapterBased) optimizerRaw = rankAdamw * policy.adapterOptimizerBytes;
      else if ((opts.optimizerType || opts.optimizer) === "muon") {
        optimizerRaw = rankMuon * 4 + rankAdamw * 12;
      } else {
        optimizerRaw = (rankMuon + rankAdamw) * policy.fullOptimizerBytes;
      }
    }

    return {
      policy,
      perRank: rankBase + rankBuffer + rankAdapter,
      rankBase,
      rankBuffer,
      rankBaseTrainable,
      rankAdapter,
      rankTrainable,
      rankFrozen,
      rankMuon,
      rankAdamw,
      weightBytes: weightRaw,
      gradBytes: gradRaw,
      optimizerBytes: optimizerRaw,
      nRankExpert,
      nRankTpShard,
      nRankReplicated,
    };
  }

  function estimateParameterMemory(options) {
    const opts = options || {};
    const baseParams = finiteNonNegative(opts.baseParams);
    if (baseParams == null) throw new Error("baseParams 必须是非负数");
    return aggregateComponentMemory(Object.assign({}, opts, {
      components: [{
        key: "base",
        numel: baseParams,
        trainable: true,
        parallel: "replicated",
        optimizer: "adamw",
        dtypeBytes: opts.trainWeightBytes || 2,
        mtpNumel: finiteNonNegative(opts.mtpParams) || 0,
      }],
      rankScale: function () {
        const rankParams = finiteNonNegative(opts.rankBaseParams);
        return baseParams > 0 && rankParams != null ? rankParams / baseParams : 1;
      },
    }));
  }

  return Object.freeze({
    DEFAULT_TRAINABLE_RATIO,
    listTrainingModes,
    findTrainingMode,
    listArtifacts,
    findArtifact,
    defaultArtifactId,
    parseArtifactAlias,
    resolveArtifact,
    computeTrainableParams,
    resolveTrainingPolicy,
    aggregateComponentMemory,
    estimateParameterMemory,
  });
});
