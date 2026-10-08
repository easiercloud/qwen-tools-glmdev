const test = require("node:test");
const assert = require("node:assert/strict");

const glm = require("../memory/glm-modeling.js");

function load(modelId) {
  const config = glm.makePresetConfig(modelId);
  assert.ok(config, "missing preset " + modelId);
  return glm.parseConfig(config, { modelId });
}

test("GLM-5.2 config maps MLA, mixed MLP, DSA sharing, and MTP", () => {
  const model = load("glm-5.2");
  assert.equal(model.architectureKind, "glm_moe_dsa");
  assert.equal(model.experts, 256);
  assert.equal(model.nSharedExperts, 1);
  assert.equal(model.qLoraRank, 2048);
  assert.equal(model.kvLoraRank, 512);
  assert.equal(model.qkNopeHeadDim, 192);
  assert.equal(model.qkRopeHeadDim, 64);
  assert.equal(model.vHeadDim, 256);
  assert.equal(model.mlpLayerTypes.filter((x) => x === "dense").length, 3);
  assert.equal(model.mlpLayerTypes.filter((x) => x === "sparse").length, 75);
  assert.equal(model.indexerTypes.filter((x) => x === "full").length, 21);
  assert.equal(model.indexerTypes.filter((x) => x === "shared").length, 57);
  assert.equal(model.mtp, 1);
});

test("GLM-5.2 component ledger reproduces checkpoint bytes exactly", () => {
  const model = load("glm-5.2");
  const summary = glm.summarizeComponents(model);
  assert.equal(summary.N, 753329940480);
  assert.equal(summary.trainableN, 753329921024);
  assert.equal(summary.bufferN, 19456);
  assert.equal(summary.checkpointBytes, 1506659919872);
  assert.equal(summary.checkpointTruthBytes, 1506659919872);
  assert.equal(summary.checkpointDeltaBytes, 0);
  assert.equal(summary.fullIndexerTotal, 22);
  assert.equal(summary.sparseTotal, 76);
});

test("GLM-5.3 and GLM-5.3-BF16 share the 5.2 architecture", () => {
  const p52 = glm.summarizeComponents(load("glm-5.2"));
  const p53 = glm.summarizeComponents(load("glm-5.3"));
  const p53bf16 = glm.summarizeComponents(load("glm-5.3-bf16"));
  assert.equal(p53.N, p52.N);
  assert.equal(p53bf16.N, p52.N);
  assert.equal(p53.checkpointTruthBytes, 755617140416);
  assert.equal(p53bf16.checkpointTruthBytes, p52.checkpointTruthBytes);
});

test("TP only shards TP components and EP only shards routed experts", () => {
  const model = load("glm-5.2");
  const base = { TP: 1, PP: 1, EP: 1, ETP: 1, bw: 2, optimizer: "muon", optimizerOffload: true };
  const one = glm.aggregateStatic(model, base);
  const tp2 = glm.aggregateStatic(model, { ...base, TP: 2 });
  const ep8 = glm.aggregateStatic(model, { ...base, EP: 8 });
  assert.equal(tp2.nRankReplicated, one.nRankReplicated);
  assert.equal(tp2.nRankExpert, one.nRankExpert);
  assert.equal(tp2.nRankTpShard, one.nRankTpShard / 2);
  assert.equal(ep8.nRankReplicated, one.nRankReplicated);
  assert.equal(ep8.nRankTpShard, one.nRankTpShard);
  assert.equal(ep8.nRankExpert, one.nRankExpert / 8);
});

test("Muon+AdamW, AdamW, and optimizer offload use distinct exact states", () => {
  const model = load("glm-5.2");
  const base = { TP: 2, PP: 1, EP: 8, ETP: 1, bw: 2 };
  const muon = glm.aggregateStatic(model, { ...base, optimizer: "muon", optimizerOffload: false });
  const adamw = glm.aggregateStatic(model, { ...base, optimizer: "adamw", optimizerOffload: false });
  const offload = glm.aggregateStatic(model, { ...base, optimizer: "muon", optimizerOffload: true });
  const expectedMuon = (muon.rankMuon * 4 + muon.rankAdamw * 12) / (1024 ** 3);
  const expectedAdamw = (adamw.rankMuon + adamw.rankAdamw) * 12 / (1024 ** 3);
  assert.equal(muon.optimizer, expectedMuon);
  assert.equal(adamw.optimizer, expectedAdamw);
  assert.ok(adamw.optimizer > muon.optimizer);
  assert.equal(offload.optimizer, 0);
});

test("CE chunking divides only the CE workspace", () => {
  const model = load("glm-5.2");
  const base = { TP: 2, mbs: 1, prompt: 1024, resp: 2048 };
  const one = glm.computeDynamic(model, { ...base, ceChunks: 1 });
  const four = glm.computeDynamic(model, { ...base, ceChunks: 4 });
  assert.equal(four.ce, one.ce / 4);
  assert.equal(four.dsaIndices, one.dsaIndices);
  assert.equal(four.routerLogits, one.routerLogits);
  assert.equal(four.total, Math.max(four.outputPhase, four.attentionPhase));
});

test("makePresetConfig applies a layer override and regenerates layer distributions", () => {
  const cfg = glm.makePresetConfig("glm-5.2", 8);
  assert.equal(cfg.num_hidden_layers, 8);
  assert.equal(cfg.mlp_layer_types.length, 8);
  assert.equal(cfg.mlp_layer_types.filter((x) => x === "dense").length, 3);
  assert.equal(cfg.mlp_layer_types.filter((x) => x === "sparse").length, 5);
  assert.equal(cfg.indexer_types.length, 8);
  assert.equal(cfg.indexer_types.filter((x) => x === "full").length, 4);
  assert.equal(cfg.num_nextn_predict_layers, 1);
});

test("layer overrides drop checkpoint truth and mark the config synthetic", () => {
  const model = glm.parseConfig(glm.makePresetConfig("glm-5.2", 16), { modelId: "glm-5.2", layerOverride: 16 });
  assert.equal(model.layers, 16);
  assert.equal(model.checkpointTruthBytes, null);
  assert.match(model.sourceRevision, /scaled/);
  const summary = glm.summarizeComponents(model);
  assert.equal(summary.checkpointDeltaBytes, null);
  assert.equal(summary.sparseTotal, 13 + 1);
  assert.equal(summary.fullIndexerTotal, 6 + 1);
  assert.ok(summary.N < 753329940480);
});

test("scaled preset ids build synthetic configs without checkpoint truth", () => {
  const cfg8 = glm.makePresetConfig("glm-5.2-8l");
  assert.equal(cfg8.num_hidden_layers, 8);
  const model8 = glm.parseConfig(cfg8, { modelId: "glm-5.2-8l" });
  assert.equal(model8.layers, 8);
  assert.equal(model8.checkpointTruthBytes, null);
  assert.match(model8.sourceRevision, /scaled/);
  const model16 = glm.parseConfig(glm.makePresetConfig("glm-5.2-16l"), { modelId: "glm-5.2-16l" });
  assert.equal(model16.layers, 16);
  assert.equal(model16.checkpointTruthBytes, null);
  assert.equal(glm.makePresetConfig("glm-5.2-99l"), null);
});

test("withLayerOverride rewrites a pasted config and leaves the original intact", () => {
  const original = glm.makePresetConfig("glm-5.2");
  const scaled = glm.withLayerOverride(original, 12);
  assert.equal(original.num_hidden_layers, 78);
  assert.equal(original.mlp_layer_types.length, 78);
  assert.equal(scaled.num_hidden_layers, 12);
  assert.equal(scaled.mlp_layer_types, undefined);
  assert.equal(scaled.indexer_types, undefined);
  const model = glm.parseConfig(scaled, { modelId: "glm-5.2", layerOverride: 12 });
  assert.equal(model.layers, 12);
  assert.equal(model.mlpLayerTypes.filter((x) => x === "dense").length, 3);
  assert.equal(model.checkpointTruthBytes, null);
});

test("layer override below the dense prefix clamps instead of throwing", () => {
  const model = glm.parseConfig(glm.makePresetConfig("glm-5.2", 2), { modelId: "glm-5.2", layerOverride: 2 });
  assert.equal(model.layers, 2);
  assert.deepEqual(model.mlpLayerTypes, ["dense", "dense"]);
  glm.summarizeComponents(model);
});

test("unsupported GLM-5.3-Flash fails closed", () => {
  assert.throws(
    () => glm.parseConfig({ text_config: { model_type: "glm5_next" } }),
    /KDA\/MHC\/ViT/
  );
});
