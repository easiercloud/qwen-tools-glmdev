const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.join(__dirname, "..");

const QWEN_27_CONFIG = {
  architectures: ["Qwen3_5ForConditionalGeneration"],
  text_config: {
    vocab_size: 248320, hidden_size: 5120, num_hidden_layers: 64,
    intermediate_size: 17408, num_attention_heads: 24,
    num_key_value_heads: 4, head_dim: 256, full_attention_interval: 4,
    linear_num_key_heads: 16, linear_key_head_dim: 128,
    linear_num_value_heads: 48, linear_value_head_dim: 128,
    linear_conv_kernel_dim: 4, attn_output_gate: true,
  },
  vision_config: {
    depth: 27, hidden_size: 1152, intermediate_size: 4304,
    out_hidden_size: 5120, patch_size: 16, temporal_patch_size: 2,
    in_channels: 3, spatial_merge_size: 2, num_position_embeddings: 2304,
  },
  tie_word_embeddings: false,
};

function makeElement(id) {
  return {
    id,
    value: "",
    checked: false,
    disabled: false,
    innerHTML: "",
    textContent: "",
    style: {},
    addEventListener() {},
  };
}

async function createPage() {
  const ids = [
    "ovModel", "ovArtifact", "ovTrainingMode", "ovLoraRank",
    "artifactField", "trainingModeField", "loraRankField", "trainingBoundary",
    "mtpTrainField", "ovMtpEnabled", "chips", "err",
    "ovP", "ovHbm", "ovWrap", "ovHccl", "ovMtp", "wrapField",
    "ovSeq", "ovMbs", "ovChunk", "ovPrefetch", "ovRecompute", "ovOffload",
    "ovSkipGdn", "ovGmm", "gmmRow", "ovFreeze", "vTotal", "vTotalSub",
    "hbmBar", "vRemain", "blocks", "mapBody", "mapFoot",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, makeElement(id)]));
  const context = {
    console,
    fetch: async () => ({ ok: true, json: async () => QWEN_27_CONFIG }),
    document: {
      getElementById(id) { return elements[id] || (elements[id] = makeElement(id)); },
      querySelectorAll() { return Object.values(elements); },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  for (const file of ["model-catalog.js", "training-policies.js", "glm-modeling.js"]) {
    vm.runInContext(fs.readFileSync(path.join(root, "memory", file), "utf8"), context);
  }
  const html = fs.readFileSync(path.join(root, "memory", "fsdp.html"), "utf8");
  const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .map((match) => match[1])
    .filter((source) => source.trim());
  for (const source of inlineScripts) vm.runInContext(source, context);
  await vm.runInContext("loadModel(MemoryModelCatalog.findModel(S.modelName), true)", context);
  return { context, elements };
}

async function selectGlm(context, elements, modelId = "glm-5.2") {
  elements.ovModel.value = modelId;
  await vm.runInContext("onModelPicked()", context);
}

test("FSDP keeps the original single Qwen selector and default", async () => {
  const { context, elements } = await createPage();
  assert.equal(elements.ovModel.value, "qwen3.5-27b");
  assert.match(elements.ovModel.innerHTML, /Qwen3\.5/);
  assert.match(elements.ovModel.innerHTML, /GLM-5\.2/);
  assert.equal(elements.artifactField.style.display, "none");
  assert.equal(elements.trainingModeField.style.display, "none");
  assert.match(elements.ovArtifact.innerHTML, /W4A8/);
  assert.match(elements.ovArtifact.innerHTML, /W8A8/);
  assert.equal(vm.runInContext("S.loadState", context), "ready");
});

test("GLM FSDP uses the exact component ledger", async () => {
  const { context, elements } = await createPage();
  await selectGlm(context, elements);
  assert.equal(elements.ovModel.value, "glm-5.2");
  assert.equal(elements.artifactField.style.display, "");
  assert.equal(vm.runInContext("S.model.family", context), "glm");
  assert.equal(vm.runInContext("compute(S.model, S.train).N", context), 753329940480);
  assert.match(elements.mapBody.innerHTML, /GLM 组件账本/);
  assert.match(elements.mapBody.innerHTML, /DSA indexer/);
});

test("Qwen FSDP default preserves the origin/main golden breakdown", async () => {
  const { context } = await createPage();
  const result = vm.runInContext("applyDenseScheme(compute(S.model, S.train), S.model)", context);
  assert.equal(result.N, 27356665200);
  assert.ok(Math.abs(result.paramG - 6.369469966739416) < 1e-12);
  assert.ok(Math.abs(result.gradG - 12.738939933478832) < 1e-12);
  assert.ok(Math.abs(result.adamG - 25.477879866957664) < 1e-12);
  assert.ok(Math.abs(result.deviceG - 57.58995187655091) < 1e-12);
  assert.equal(result.policy.fullOptimizerBytes, 8);
});

test("Qwen freeze-ViT retains the origin/main exclusion semantics", async () => {
  const { context } = await createPage();
  const full = vm.runInContext("compute(S.model, { ...S.train, freezeVisual: false })", context);
  const frozen = vm.runInContext("compute(S.model, { ...S.train, freezeVisual: true })", context);
  assert.ok(full.vit > 0);
  assert.equal(frozen.vit, 0);
  assert.equal(frozen.N, full.N - full.vit);
  assert.ok(frozen.paramG < full.paramG);
  assert.ok(frozen.gradG < full.gradG);
  assert.equal(frozen.policy.fullOptimizerBytes, 8);
});

test("Qwen 20L MoE keeps its page-local architecture and Qwen optimizer policy", async () => {
  const { context, elements } = await createPage();
  elements.ovModel.value = "qwen3.6-35b-a3b-20l";
  await vm.runInContext("onModelPicked()", context);
  const result = vm.runInContext("compute(S.model, S.train)", context);
  assert.equal(result.N, 18285373616);
  assert.equal(result.L, 20);
  assert.equal(result.policy.fullOptimizerBytes, 8);
  assert.equal(elements.artifactField.style.display, "none");
  assert.equal(elements.trainingModeField.style.display, "none");
});

test("LoRA and QLoRA only train adapters while the frozen base stays resident", async () => {
  const { context, elements } = await createPage();
  await selectGlm(context, elements);
  const full = vm.runInContext("compute(S.model, { ...S.train, trainingMode: 'full_sft' })", context);
  const lora = vm.runInContext("compute(S.model, { ...S.train, trainingMode: 'lora', loraRank: 16 })", context);
  vm.runInContext("S.artifactId = 'w4a8-c8'", context);
  const qlora = vm.runInContext("compute(S.model, { ...S.train, trainingMode: 'qlora', loraRank: 16 })", context);
  assert.ok(lora.parameterMemory.rankFrozen > 0);
  assert.ok(lora.parameterMemory.rankAdapter > 0);
  assert.ok(lora.gradG < full.gradG);
  assert.ok(lora.adamG < full.adamG);
  assert.ok(qlora.paramG < lora.paramG);
  assert.equal(qlora.policy.artifact.artifactId, "w4a8-c8");
});

test("selecting QLoRA automatically chooses a low-bit GLM artifact", async () => {
  const { context, elements } = await createPage();
  await selectGlm(context, elements);
  assert.equal(elements.ovArtifact.value, "bf16");
  elements.ovTrainingMode.value = "qlora";
  vm.runInContext("onTrainingModePicked()", context);
  assert.equal(elements.ovArtifact.value, "w4a8-c8");
  assert.match(elements.err.textContent, /未验证可加载性/);
});

test("GLM MTP structure stays resident while its training state can be disabled", async () => {
  const { context, elements } = await createPage();
  await selectGlm(context, elements);
  const enabled = vm.runInContext("compute(S.model, S.train)", context);
  elements.ovMtp.value = "0";
  elements.ovMtpEnabled.checked = false;
  vm.runInContext("readForm(); render()", context);
  const disabled = vm.runInContext("compute(S.model, S.train)", context);
  assert.equal(vm.runInContext("S.model.mtp", context), 1);
  assert.equal(disabled.N, enabled.N);
  assert.equal(disabled.paramG, enabled.paramG);
  assert.ok(disabled.gradG < enabled.gradG);
  assert.ok(disabled.adamG < enabled.adamG);
});

test("reinforcement learning is explicitly scoped to actor update", async () => {
  const { context, elements } = await createPage();
  await selectGlm(context, elements);
  elements.ovTrainingMode.value = "rl_actor";
  vm.runInContext("onTrainingModePicked()", context);
  assert.match(elements.trainingBoundary.textContent, /actor update/);
  assert.match(elements.trainingBoundary.textContent, /rollout/);
  assert.match(elements.vTotalSub.textContent, /强化学习/);
});
