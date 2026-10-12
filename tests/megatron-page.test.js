const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.join(__dirname, "..");

function makeElement(id) {
  return {
    id,
    value: "",
    checked: false,
    hidden: false,
    disabled: false,
    title: "",
    innerHTML: "",
    textContent: "",
    style: {},
    tagName: "DIV",
    addEventListener() {},
    closest() { return null; },
  };
}

function createPage(search) {
  const ids = [
    "ovModel", "chips", "err", "ovTP", "ovPP", "ovEP", "ovETP",
    "ovHbm", "ovHccl", "ovMtp", "ovLayers", "ovMbs", "ovPrompt", "ovResp", "ovCeChunks",
    "ovOptimizer", "ovAdamOff", "ovRecompute", "ovEntropy", "epField", "etpField",
    "ovArtifact", "ovArtifactAlias", "ovTrainMode", "ovTrainablePct", "ovTrainableParams", "ovMtpEnabled",
    "artifactField", "artifactAliasField", "trainingModeCard", "customCfgBox", "layerOverrideField",
    "ceChunksField", "optimizerField", "customCfg", "customCfgLoad",
    "vTotal", "vTotalSub", "hbmBar", "vRemain", "blocks", "mapBody", "mapFoot",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, makeElement(id)]));
  const context = {
    console,
    URL,
    URLSearchParams,
    location: {
      search: search || "",
      href: "http://127.0.0.1:8876/memory/megatron.html" + (search || ""),
    },
    history: {
      replaceState(_state, _title, url) { context.location.href = String(url); },
    },
    fetch: async () => { throw new Error("network disabled by page test"); },
    document: {
      getElementById(id) { return elements[id] || (elements[id] = makeElement(id)); },
      querySelectorAll() { return Object.values(elements); },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, "memory/model-catalog.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(root, "memory/training-policies.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(root, "memory/glm-modeling.js"), "utf8"), context);
  const html = fs.readFileSync(path.join(root, "memory/megatron.html"), "utf8");
  const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map((match) => match[1])
    .filter((source) => source.trim());
  for (const source of inlineScripts) vm.runInContext(source, context);
  return { context, elements };
}

async function selectModel(context, elements, modelId) {
  elements.ovModel.value = modelId;
  await vm.runInContext("onModelPicked()", context);
}

test("page preserves the origin/main Qwen default and golden memory", () => {
  const { context, elements } = createPage();
  assert.equal(elements.ovModel.value, "qwen3.5-27b");
  assert.equal(elements.ovMtp.value, "0");
  assert.equal(elements.ovMtp.disabled, false);
  assert.equal(elements.ovArtifact.value, "bf16");
  assert.equal(elements.ovTrainMode.value, "pretrain");
  assert.equal(elements.ovOptimizer.value, "adam");
  assert.equal(elements.ovLayers.disabled, true);
  assert.equal(elements.artifactField.style.display, "none");
  assert.equal(elements.trainingModeCard.style.display, "none");
  const result = vm.runInContext("compute(S.model, S.train)", context);
  assert.equal(result.N, 27356665200);
  assert.ok(Math.abs(result.W - 12.738939933478832) < 1e-12);
  assert.ok(Math.abs(result.G - 12.738939933478832) < 1e-12);
  assert.ok(Math.abs(result.dynG - 1.4208984375) < 1e-12);
  assert.ok(Math.abs(result.deviceG - 33.180028304457664) < 1e-12);
});

test("Qwen keeps the origin/main TP and PP validation semantics", () => {
  const { context } = createPage();
  const result = vm.runInContext("compute(S.model, { ...S.train, TP: 3, PP: 2 })", context);
  assert.equal(result.tpIllegal, false);
  assert.equal(result.peakOk, true);
  assert.ok(!result.warns.some((warning) => /PP>1/.test(warning)));
});

test("ordinary Qwen selection does not rewrite its legacy deep-link URL", async () => {
  const { context, elements } = createPage("?model=Qwen3.5-27B");
  const before = context.location.href;
  elements.ovModel.value = "qwen3.5-27b";
  await vm.runInContext("onModelPicked()", context);
  assert.equal(context.location.href, before);
});

test("legacy label deep links resolve to the new stable model id", () => {
  const { elements } = createPage("?model=GLM-5.3-BF16");
  assert.equal(elements.ovModel.value, "glm-5.3");
  assert.equal(elements.ovArtifact.value, "bf16");
  assert.match(elements.chips.innerHTML, /GLM/);
  assert.match(elements.chips.innerHTML, /BF16 artifact/);
});

test("GLM-5.3 selection exposes FP8 checkpoint but keeps BF16 training semantics", async () => {
  const { context, elements } = createPage();
  elements.ovModel.value = "glm-5.3";
  await vm.runInContext("onModelPicked()", context);
  assert.equal(elements.ovModel.value, "glm-5.3");
  assert.equal(elements.ovArtifact.value, "fp8");
  assert.match(elements.chips.innerHTML, /FP8 artifact/);
  assert.match(elements.err.textContent, /全参更新已按训练 dtype 展开/);
  assert.equal(vm.runInContext("S.train.bw", context), 2);
  assert.equal(vm.runInContext("countParams(S.model).N", context), 753329940480);
});

test("switching to the local Qwen preset preserves its prior golden count", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  await selectModel(context, elements, "qwen3.5-27b");
  assert.equal(elements.ovModel.value, "qwen3.5-27b");
  assert.equal(elements.ovMtp.disabled, false);
  assert.equal(elements.ovOptimizer.value, "adam");
  assert.equal(vm.runInContext("countParams(S.model).N", context), 27356665200);
});

test("local Qwen MoE preset preserves its prior golden count", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "qwen3.6-35b-a3b-20l");
  assert.equal(vm.runInContext("countParams(S.model).N", context), 18285373616);
  assert.match(elements.chips.innerHTML, /MOE/);
});

test("remote Qwen load failure preserves the previous result", async () => {
  const { context, elements } = createPage();
  const failed = await vm.runInContext(
    "loadModel(MemoryModelCatalog.findModel('qwen3-8b'), true)",
    context
  );
  assert.equal(failed, undefined);
  assert.equal(vm.runInContext("S.loadState", context), "error");
  assert.equal(vm.runInContext("S.loadedModelId", context), null);
  assert.match(elements.err.textContent, /拉取 Qwen\/Qwen3-8B/);
  assert.match(elements.vTotal.innerHTML, /^33\.180/);
  assert.notEqual(elements.mapBody.innerHTML, "");
});

test("layer override input rebuilds the GLM model and keeps training params", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  elements.ovLayers.value = "8";
  await vm.runInContext("onLayersInput()", context);
  assert.equal(vm.runInContext("S.model.layers", context), 8);
  assert.equal(vm.runInContext("S.model.checkpointTruthBytes", context), null);
  assert.equal(vm.runInContext("S.loadedModelId", context), "glm-5.2");
  assert.equal(vm.runInContext("S.train.TP", context), 4);
  assert.ok(vm.runInContext("countParams(S.model).N", context) < 753329940480);
  assert.match(elements.chips.innerHTML, /合成减层配置/);
  assert.equal(elements.ovLayers.value, "8");
});

test("small-layer validation is not a model option and still uses the layer override", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  assert.doesNotMatch(elements.ovModel.innerHTML, /glm-5\.2-8l/);
  elements.ovLayers.value = "8";
  await vm.runInContext("onLayersInput()", context);
  assert.equal(vm.runInContext("S.model.layers", context), 8);
  assert.equal(vm.runInContext("S.model.checkpointTruthBytes", context), null);
  assert.equal(vm.runInContext("S.loadState", context), "ready");
  assert.equal(vm.runInContext("S.customConfig", context), null);
  assert.equal(elements.ovLayers.value, "8");
  assert.match(elements.chips.innerHTML, /合成减层配置/);
});

test("custom GLM config paste builds a synthetic model and supports re-scaling", async () => {
  const { context, elements } = createPage();
  await vm.runInContext(`
    (async () => {
      $("customCfg").value = JSON.stringify(GlmMemoryModel.makePresetConfig("glm-5.2", 6));
      onCustomConfigLoad();
      $("ovLayers").value = "10";
      onLayersInput();
    })()
  `, context);
  assert.equal(vm.runInContext("S.modelName", context), "custom-config");
  assert.equal(vm.runInContext("S.customConfig.num_hidden_layers", context), 6);
  assert.equal(vm.runInContext("S.model.layers", context), 10);
  assert.equal(vm.runInContext("S.model.checkpointTruthBytes", context), null);
  assert.equal(vm.runInContext("S.loadState", context), "ready");
  assert.match(elements.ovModel.innerHTML, /custom-config/);
  assert.equal(elements.ovLayers.value, "10");
});

test("custom config load reports invalid JSON and keeps the previous model", async () => {
  const { context, elements } = createPage();
  await vm.runInContext(`
    (async () => {
      $("customCfg").value = "{not json";
      onCustomConfigLoad();
    })()
  `, context);
  assert.match(elements.err.textContent, /解析失败/);
  assert.equal(vm.runInContext("S.modelName", context), "qwen3.5-27b");
  assert.equal(vm.runInContext("S.model.layers", context), 64);
});

test("non-GLM models disable the layer override input", async () => {
  const { context, elements } = createPage();
  assert.equal(elements.ovLayers.disabled, true);
  assert.equal(elements.ovLayers.value, "");
});

test("artifact selection is independent from the GLM release", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  elements.ovArtifact.value = "w4a8-c8";
  await vm.runInContext("onArtifactPicked()", context);
  assert.equal(elements.ovModel.value, "glm-5.2");
  assert.equal(elements.ovArtifact.value, "w4a8-c8");
  assert.equal(vm.runInContext("S.model.artifactId", context), "w4a8-c8");
  assert.match(elements.chips.innerHTML, /W4A8/);
});

test("QLoRA freezes the base and allocates gradients only for adapters", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  elements.ovTrainMode.value = "qlora";
  elements.ovTrainablePct.value = "0.1";
  await vm.runInContext("onTrainingModePicked()", context);
  const result = vm.runInContext("compute(S.model, S.train)", context);
  assert.equal(elements.ovArtifact.value, "w4a8-c8");
  assert.equal(result.policy.modeId, "qlora");
  assert.ok(result.rankAdapter > 0);
  assert.ok(result.rankFrozen > result.rankAdapter);
  assert.ok(result.G < result.W);
});

test("raw artifact alias retains hardware and revision without toggling MTP", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  elements.ovArtifactAlias.value = "GLM-5.2-W4A8C8-A5-0731";
  await vm.runInContext("onArtifactAliasPicked()", context);
  assert.equal(elements.ovArtifact.value, "w4a8-c8");
  assert.equal(vm.runInContext("S.model.artifact.hardware", context), "A5");
  assert.equal(vm.runInContext("S.model.artifact.revision", context), "0731");
  assert.equal(elements.ovMtpEnabled.checked, true);
  assert.match(elements.chips.innerHTML, /硬件 A5/);
});

test("MTP training is an independent switch from checkpoint MTP parameters", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  const enabled = vm.runInContext("compute(S.model, S.train).rankTrainable", context);
  elements.ovMtpEnabled.checked = false;
  vm.runInContext("readForm(); render()", context);
  const disabled = vm.runInContext("compute(S.model, S.train).rankTrainable", context);
  assert.equal(vm.runInContext("S.model.mtp", context), 1);
  assert.ok(disabled < enabled);
  assert.match(elements.chips.innerHTML, /MTP 训练 OFF/);
});

test("leaving RL restores the entropy setting that preceded RL", async () => {
  const { context, elements } = createPage();
  await selectModel(context, elements, "glm-5.2");
  elements.ovEntropy.checked = false;
  elements.ovTrainMode.value = "rl_actor";
  vm.runInContext("onTrainingModePicked()", context);
  assert.equal(elements.ovEntropy.checked, true);
  elements.ovTrainMode.value = "full_sft";
  vm.runInContext("onTrainingModePicked()", context);
  assert.equal(elements.ovEntropy.checked, false);
});
