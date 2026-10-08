const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const catalogPath = path.join(__dirname, "..", "memory", "model-catalog.js");
const catalog = require(catalogPath);

const CURRENT_QWEN_LABELS = [
  "Qwen3.8-27B",
  "Qwen3.8-2.4T-A95B",
  "Qwen3.6-27B",
  "Qwen3.6-35B-A3B",
  "Qwen3.6-35B-A3B (20L)",
  "Qwen3.5-35B-A3B (12L)",
  "Qwen3.5-0.8B",
  "Qwen3.5-2B",
  "Qwen3.5-4B",
  "Qwen3.5-9B",
  "Qwen3.5-27B",
  "Qwen3.5-35B-A3B",
  "Qwen3.5-122B-A10B",
  "Qwen3.5-397B-A17B",
  "Qwen3-VL-8B-Instruct",
  "Qwen3-VL-32B-Instruct",
  "Qwen3-VL-30B-A3B-Instruct",
  "Qwen3-VL-235B-A22B-Instruct",
  "Qwen3-0.6B",
  "Qwen3-1.7B",
  "Qwen3-4B",
  "Qwen3-8B",
  "Qwen3-14B",
  "Qwen3-32B",
  "Qwen3-4B-Instruct",
  "Qwen3-8B-Instruct",
  "Qwen3-14B-Instruct",
  "Qwen3-32B-Instruct",
  "Qwen3-30B-A3B",
  "Qwen3-30B-A3B-Thinking-2507",
  "Qwen3-235B-A22B",
  "Qwen3-235B-A22B-Instruct-2507",
  "Qwen3-Next-80B-A3B-Instruct",
  "Qwen3-Coder-30B-A3B-Instruct",
  "Qwen3-Coder-480B-A35B-Instruct",
  "Qwen2.5-VL-7B-Instruct",
  "Qwen2.5-VL-72B-Instruct",
];

test("exports the catalog through CommonJS and browser UMD", () => {
  assert.equal(typeof catalog.listSeries, "function");
  assert.equal(typeof catalog.listModels, "function");
  assert.equal(typeof catalog.findModel, "function");
  assert.equal(typeof catalog.findSeriesForModel, "function");

  const source = fs.readFileSync(catalogPath, "utf8");
  const browserContext = {};
  vm.createContext(browserContext);
  vm.runInContext(source, browserContext);
  assert.equal(typeof browserContext.MemoryModelCatalog.listModels, "function");
  assert.equal(browserContext.MemoryModelCatalog.defaultModelId, catalog.defaultModelId);
});

test("series and model ids are unique and every model belongs to a series", () => {
  const series = catalog.listSeries();
  const models = catalog.listModels();
  assert.equal(new Set(series.map((item) => item.seriesId)).size, series.length);
  assert.equal(new Set(models.map((item) => item.modelId)).size, models.length);
  assert.equal(new Set(models.map((item) => item.label)).size, models.length);

  const seriesIds = new Set(series.map((item) => item.seriesId));
  for (const model of models) {
    assert.ok(seriesIds.has(model.seriesId), model.modelId + " has an unknown series");
    assert.equal(model.seriesLabel, catalog.findSeriesForModel(model.modelId).seriesLabel);
  }
});

test("contains every Qwen option currently exposed by the HTML pages", () => {
  const actual = catalog.listModels()
    .filter((model) => model.capabilities.vendor === "qwen")
    .map((model) => model.label)
    .sort();
  assert.deepEqual(actual, CURRENT_QWEN_LABELS.slice().sort());
});

test("maps repositories and preserves the three local Qwen preset sources", () => {
  const localPresets = catalog.listModels()
    .filter((model) => model.source.startsWith("local:"))
    .map((model) => [model.modelId, model.repo, model.source])
    .sort();

  assert.deepEqual(localPresets, [
    ["qwen3.5-27b", "Qwen/Qwen3.5-27B", "local:27B"],
    ["qwen3.5-35b-a3b-12l", "Qwen/Qwen3.5-35B-A3B", "local:12L"],
    ["qwen3.6-35b-a3b-20l", "Qwen/Qwen3.6-35B-A3B", "local:20L"],
  ]);

  assert.equal(catalog.findModel("qwen3-8b").repo, "Qwen/Qwen3-8B");
  assert.equal(catalog.findModel("glm-5.2").repo, "ZhipuAI/GLM-5.2");
  assert.equal(catalog.findModel("glm-5.3").repo, "ZhipuAI/GLM-5.3");
  assert.equal(catalog.findModel("glm-5.3-bf16").repo, "ZhipuAI/GLM-5.3-BF16");
  assert.equal(catalog.findModel("glm-5.3-flash").repo, "ZhipuAI/GLM-5.3-Flash");
});

test("filters models by series and resolves ids and legacy labels", () => {
  const glmModels = catalog.listModels("glm-5");
  assert.deepEqual(glmModels.map((model) => model.modelId), [
    "glm-5.2",
    "glm-5.3",
    "glm-5.3-bf16",
    "glm-5.2-8l",
    "glm-5.2-16l",
    "glm-5.3-flash",
  ]);
  assert.ok(glmModels.every((model) => model.seriesId === "glm-5"));
  assert.deepEqual(catalog.listModels("not-a-series"), []);
  assert.equal(catalog.findModel("GLM-5.2").modelId, "glm-5.2");
  assert.equal(catalog.findModel("not-a-model"), null);
  assert.equal(catalog.findSeriesForModel("glm-5.2").seriesId, "glm-5");
  assert.equal(catalog.findSeriesForModel("not-a-model"), null);
});

test("declares valid global and per-series defaults", () => {
  assert.equal(catalog.defaultModelId, "qwen3.5-27b");
  assert.ok(catalog.findModel(catalog.defaultModelId));

  for (const series of catalog.listSeries()) {
    const defaultModel = catalog.findModel(series.defaultModelId);
    assert.ok(defaultModel, series.seriesId + " has no default model");
    assert.equal(defaultModel.seriesId, series.seriesId);
    assert.equal(defaultModel.supported, true, series.seriesId + " defaults to an unsupported model");
  }
});

test("marks GLM-5.3-Flash as an unsupported independent architecture", () => {
  for (const modelId of ["glm-5.2", "glm-5.3", "glm-5.3-bf16"]) {
    assert.equal(catalog.findModel(modelId).supported, true);
  }

  const flash = catalog.findModel("glm-5.3-flash");
  assert.equal(flash.supported, false);
  assert.equal(flash.capabilities.independentArchitecture, true);
  assert.equal(flash.capabilities.architecture, "glm5-next");
  assert.match(flash.supportNote, /独立.*架构/);
});

test("exposes scaled GLM-5.2 presets for small-cluster verification", () => {
  const expected = [
    ["glm-5.2-8l", "glm-scale:8", "GLM-5.2 (8L)"],
    ["glm-5.2-16l", "glm-scale:16", "GLM-5.2 (16L)"],
  ];
  for (const [modelId, source, label] of expected) {
    const entry = catalog.findModel(modelId);
    assert.equal(entry.supported, true);
    assert.equal(entry.source, source);
    assert.equal(entry.label, label);
    assert.equal(entry.seriesId, "glm-5");
    assert.equal(entry.repo, "ZhipuAI/GLM-5.2");
    assert.equal(entry.capabilities.vendor, "zhipuai");
    assert.equal(entry.capabilities.architecture, "glm-moe-dsa");
    assert.match(entry.supportNote, /合成/);
  }
});
