# 模型训练小工具

面向 Qwen、GLM 等模型的训练工程工具台。一个仓库、一个入口，按工作面拆成独立模块，避免显存、数据、性能的改动互相踩。

在线入口：[zyang6.github.io/qwen-tools](https://zyang6.github.io/qwen-tools/)  
源码：[github.com/zyang6/qwen-tools](https://github.com/zyang6/qwen-tools)

## 模块

| 编号 | 工作面 | 目录 | 状态 |
|---|---|---|---|
| 01 | 显存分析 | [`memory/`](memory/) | Megatron：Qwen + GLM-5.2/5.3；FSDP / 推理保持原能力 |
| 02 | 数据 | [`data/`](data/) | 空 |
| 03 | 性能 | [`perf/`](perf/) | 空 |
| 04 | 算子 | [`ops/`](ops/) | 空 |
| 05 | 已知问题 | [`issues/`](issues/) | 空 |
| 06 | 技术报告 | [`reports/`](reports/) | 空 |

## 仓库结构

```text
qwen-tools/
├── index.html          # 总入口（GitHub Pages）
├── README.md           # 本文件：模块地图
├── memory/             # 01 显存分析
│   ├── README.md
│   ├── megatron.html
│   ├── model-catalog.js
│   ├── glm-modeling.js
│   ├── fsdp.html
│   └── infer.html
├── data/               # 02 数据
├── perf/               # 03 性能
├── ops/                # 04 算子
├── issues/             # 05 已知问题
└── reports/            # 06 技术报告
```

## 本地预览

```bash
cd qwen-tools
python3 -m http.server 8876 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8876/`。

## 回归测试

```bash
node --test tests/*.test.js
```
