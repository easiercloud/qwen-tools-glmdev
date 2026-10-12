const test = require("node:test");
const assert = require("node:assert/strict");

const policies = require("../memory/training-policies.js");

test("exposes architecture-independent artifact and training mode presets", () => {
  assert.deepEqual(policies.listTrainingModes().map((mode) => mode.modeId), [
    "pretrain", "full_sft", "lora", "qlora", "rl_actor",
  ]);
  assert.deepEqual(policies.listArtifacts("qwen3-8b").map((item) => item.artifactId), [
    "bf16", "fp8", "w8a8-c8", "w4a8-c8", "w4a4c8-mxfp4", "w8a8c8-mxfp8",
  ]);
  assert.equal(policies.defaultArtifactId("glm-5.2"), "bf16");
  assert.equal(policies.defaultArtifactId("glm-5.3"), "fp8");
});

test("normalizes raw aliases while retaining hardware, revision, and MTP metadata", () => {
  const parsed = policies.parseArtifactAlias("GLM-5.2-W4A8C8-A5-0731&#x20;");
  assert.equal(parsed.modelId, "glm-5.2");
  assert.equal(parsed.artifactId, "w4a8-c8");
  assert.equal(parsed.hardware, "A5");
  assert.equal(parsed.revision, "0731");
  const mtp = policies.parseArtifactAlias("GLM-5.2-W4A8C8-0702-MTP");
  assert.equal(mtp.mtpTagged, true);
  const gpu = policies.parseArtifactAlias("GLM-5.1-W8A8 GPU-WEIGHT-0805");
  assert.equal(gpu.hardware, "GPU");
  assert.equal(gpu.revision, "0805");
  assert.equal(policies.parseArtifactAlias("GLM-5.2-FP8-DFlash").role, "drafter");
  assert.equal(policies.parseArtifactAlias("GLM-5.3-Flash-BF16").independentArchitecture, true);
});

test("resolves exact checkpoint truth separately from architecture", () => {
  assert.equal(policies.resolveArtifact({ modelId: "glm-5.2", artifactId: "bf16" }).checkpointBytes, 1506659919872);
  assert.equal(policies.resolveArtifact({ modelId: "glm-5.2", artifactId: "fp8" }).checkpointBytes, 755617140416);
  assert.equal(policies.resolveArtifact({ modelId: "glm-5.3", artifactId: "fp8" }).checkpointBytes, 755617140416);
  assert.equal(policies.resolveArtifact({ modelId: "glm-5.2", artifactId: "w4a8-c8" }).checkpointBytes, null);
});

test("full update ignores artifact storage while LoRA and QLoRA freeze the base", () => {
  const component = [{
    key: "base", numel: 1_000_000, trainable: true,
    parallel: "replicated", optimizer: "adamw", dtypeBytes: 2, mtpNumel: 100_000,
  }];
  const full = policies.aggregateComponentMemory({
    components: component,
    mode: "full_sft",
    artifactId: "w4a8-c8",
    trainWeightBytes: 2,
    optimizerType: "adam",
  });
  assert.equal(full.weightBytes, 2_000_000);
  assert.equal(full.gradBytes, 2_000_000);
  assert.equal(full.rankFrozen, 0);

  const lora = policies.aggregateComponentMemory({
    components: component,
    mode: "lora",
    artifactId: "bf16",
    trainableRatio: 0.01,
    optimizerType: "adam",
  });
  assert.equal(lora.rankAdapter, 10_000);
  assert.equal(lora.rankFrozen, 1_000_000);
  assert.equal(lora.gradBytes, 20_000);

  const qlora = policies.aggregateComponentMemory({
    components: component,
    mode: "qlora",
    artifactId: "w4a8-c8",
    trainableParams: 2_000,
    optimizerType: "adam",
  });
  assert.equal(qlora.rankAdapter, 2_000);
  assert.equal(qlora.gradBytes, 4_000);
  assert.ok(qlora.weightBytes < lora.weightBytes);
});

test("MTP switch freezes only MTP-owned parameters during a full update", () => {
  const result = policies.aggregateComponentMemory({
    components: [{
      key: "mixed", numel: 1000, mtpNumel: 100,
      trainable: true, parallel: "replicated", optimizer: "adamw", dtypeBytes: 2,
    }],
    mode: "pretrain",
    artifactId: "bf16",
    mtpEnabled: false,
    optimizerType: "adam",
  });
  assert.equal(result.rankBaseTrainable, 900);
  assert.equal(result.rankFrozen, 100);
  assert.equal(result.gradBytes, 1800);
  assert.equal(result.weightBytes, 2000);
});
