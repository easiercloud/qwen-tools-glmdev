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
    "ovSeries", "ovModel", "chips", "err", "ovTP", "ovPP", "ovEP", "ovETP",
    "ovHbm", "ovHccl", "ovMtp", "ovLayers", "ovMbs", "ovPrompt", "ovResp", "ovCeChunks",
    "ovOptimizer", "ovAdamOff", "ovRecompute", "ovEntropy", "epField", "etpField",
    "customCfg", "customCfgLoad",
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
  vm.runInContext(fs.readFileSync(path.join(root, "memory/glm-modeling.js"), "utf8"), context);
  const html = fs.readFileSync(path.join(root, "memory/megatron.html"), "utf8");
  const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map((match) => match[1])
    .filter((source) => source.trim());
  for (const source of inlineScripts) vm.runInContext(source, context);
  return { context, elements };
}

test("page defaults to GLM-5.2 with model truth and GLM optimization defaults", () => {
  const { context, elements } = createPage();
  assert.equal(elements.ovSeries.value, "glm-5");
  assert.equal(elements.ovModel.value, "glm-5.2");
  assert.equal(elements.ovMtp.value, "1");
  assert.equal(elements.ovMtp.disabled, true);
  assert.equal(elements.ovOptimizer.value, "muon");
  assert.equal(elements.ovCeChunks.value, "4");
  assert.equal(elements.ovLayers.value, "78");
  assert.equal(elements.ovLayers.disabled, false);
  assert.match(elements.chips.innerHTML, /MLA \+ DSA/);
  assert.match(elements.chips.innerHTML, /BF16 checkpoint/);
  assert.match(elements.mapBody.innerHTML, /753,329,940,480/);
  assert.equal(vm.runInContext("countParams(S.model).checkpointDeltaBytes", context), 0);
});

test("legacy label deep links resolve to the new stable model id", () => {
  const { elements } = createPage("?model=GLM-5.3-BF16");
  assert.equal(elements.ovSeries.value, "glm-5");
  assert.equal(elements.ovModel.value, "glm-5.3-bf16");
  assert.match(elements.chips.innerHTML, /GLM/);
  assert.match(elements.chips.innerHTML, /BF16 checkpoint/);
});

test("GLM-5.3 selection exposes FP8 checkpoint but keeps BF16 training semantics", async () => {
  const { context, elements } = createPage();
  elements.ovModel.value = "glm-5.3";
  await vm.runInContext("onModelPicked()", context);
  assert.equal(elements.ovModel.value, "glm-5.3");
  assert.match(elements.chips.innerHTML, /FP8 mixed checkpoint/);
  assert.match(elements.err.textContent, /训练权重\/梯度仍按 BF16/);
  assert.equal(vm.runInContext("S.train.bw", context), 2);
  assert.equal(vm.runInContext("countParams(S.model).N", context), 753329940480);
});

test("switching to the local Qwen preset preserves its prior golden count", async () => {
  const { context, elements } = createPage();
  elements.ovSeries.value = "qwen-3.5";
  await vm.runInContext("onSeriesPicked()", context);
  assert.equal(elements.ovModel.value, "qwen3.5-27b");
  assert.equal(elements.ovMtp.disabled, false);
  assert.equal(elements.ovOptimizer.value, "adamw");
  assert.equal(vm.runInContext("countParams(S.model).N", context), 27356665200);
});

test("local Qwen MoE preset preserves its prior golden count", async () => {
  const { context, elements } = createPage();
  elements.ovSeries.value = "qwen-3.6";
  await vm.runInContext("onSeriesPicked()", context);
  elements.ovModel.value = "qwen3.6-35b-a3b-20l";
  await vm.runInContext("onModelPicked()", context);
  assert.equal(vm.runInContext("countParams(S.model).N", context), 18285373616);
  assert.match(elements.chips.innerHTML, /MOE/);
});

test("remote load failure clears the previous model result", async () => {
  const { context, elements } = createPage();
  const failed = await vm.runInContext(
    "loadModel(MemoryModelCatalog.findModel('qwen3-8b'), true)",
    context
  );
  assert.equal(failed, undefined);
  assert.equal(vm.runInContext("S.loadState", context), "error");
  assert.equal(vm.runInContext("S.loadedModelId", context), null);
  assert.match(elements.err.textContent, /拉取 Qwen\/Qwen3-8B/);
  assert.match(elements.vTotal.innerHTML, /^—/);
  assert.equal(elements.mapBody.innerHTML, "");
});

test("layer override input rebuilds the GLM model and keeps training params", async () => {
  const { context, elements } = createPage();
  elements.ovLayers.value = "8";
  await vm.runInContext("onLayersInput()", context);
  assert.equal(vm.runInContext("S.model.layers", context), 8);
  assert.equal(vm.runInContext("S.model.checkpointTruthBytes", context), null);
  assert.equal(vm.runInContext("S.loadedModelId", context), "glm-5.2");
  assert.equal(vm.runInContext("S.train.TP", context), 4);
  assert.ok(vm.runInContext("countParams(S.model).N", context) < 753329940480);
  assert.match(elements.chips.innerHTML, /合成配置/);
  assert.equal(elements.ovLayers.value, "8");
});

test("scaled GLM catalog entry loads a synthetic config without remote fetch", async () => {
  const { context, elements } = createPage();
  elements.ovModel.value = "glm-5.2-8l";
  await vm.runInContext("onModelPicked()", context);
  assert.equal(vm.runInContext("S.model.layers", context), 8);
  assert.equal(vm.runInContext("S.model.checkpointTruthBytes", context), null);
  assert.equal(vm.runInContext("S.loadState", context), "ready");
  assert.equal(vm.runInContext("S.customConfig", context), null);
  assert.equal(elements.ovLayers.value, "8");
  assert.match(elements.chips.innerHTML, /合成配置/);
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
  assert.equal(vm.runInContext("S.modelName", context), "glm-5.2");
  assert.equal(vm.runInContext("S.model.layers", context), 78);
});

test("non-GLM models disable the layer override input", async () => {
  const { context, elements } = createPage();
  elements.ovSeries.value = "qwen-3.5";
  await vm.runInContext("onSeriesPicked()", context);
  assert.equal(elements.ovLayers.disabled, true);
  assert.equal(elements.ovLayers.value, "");
});
