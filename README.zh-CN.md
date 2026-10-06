# jev-laya-dsh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ci](https://github.com/alexdoandev/jev-laya-dsh/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Node](https://img.shields.io/badge/node-%E2%89%A520-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-ONNX%20%7C%20torch-blue)

<p align="center">
  <a href="README.md">English</a> | 简体中文 | <a href="README.vi.md">Tiếng Việt</a>
</p>

**jev-laya-dsh** 在 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
智能体之前加入一个本地 **System-1 决策层**，遵循 `User → Jev(Laya) → Agent` 模式：

```
User
 │ prompt
 ▼
[1] ROUTER — 事件 agent/pre-step
 │  Laya 单次前向完成 prompt 分类（task_type / needs_code_model / urgency），
 │  融合后的结论以建议性 notice 注入
 ▼
[2] GATE — 事件 tools/pre-execute（可选）
 │  对高风险工具调用做 3 探针 noul 投票；命中则转人工确认
 ▼
[3] AGENT — 大模型执行真正的工作
 │  可选：经事件 agent/request 按 tier 切换模型
 ▼
 回答
```

决策来自 **Laya** —— 开源、兼容 Jev 的校准决策模型（mmBERT 多语编码器，100+ 语言）：
不生成文本、无幻觉置信度、单次前向、$0/token、100% 本地。大模型调用只留给真正
需要它的工作。

## 特性

- **Prompt 路由** —— 在主模型消耗第一个 token 之前完成分类（chat / code /
  architecture），结论以建议性 notice 注入。
- **风险闸门** —— 对 `bash`/`pwsh` 调用做 3 探针校准投票（破坏性 / 不可逆 / 越出工作区），
  单批前向完成；触发即转人工确认而非静默执行。
- **按层切换模型** —— 架构/规划类难题（tier 2）可经 `agent/request` 整回合固定到强模型
  （RouteLLM / Hybrid LLM 式成本路由）。
- **混合驻留服务** —— ONNX 运行时保持热态（冷启动 2.5 秒、热态 0.3 秒），空闲或内存
  压力下自动卸载权重，并在会话出现时预测唤醒。
- **构造上即 fail-open** —— 每一层退化均为"本轮无路由"，决策层永不做单点故障。
- **决策 0 云端 token** —— 路由、闸门与评测全部本地运行；大模型只为主要工作计费。

## 环境要求

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（具备 Cordis 插件
  API 的任意部署）用于路由插件 —— 两个服务本身可独立运行。
- Node ≥ 20（ONNX 服务、评测、冒烟测试）。
- Python 3.12+ 与 torch —— 仅一次性 ONNX 导出需要。
- 模型权重：[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  （Apache-2.0，约 1.3 GB）。测试平台为 macOS arm64；任何可运行 onnxruntime-node 的
  环境应均可。

## 安装

```bash
git clone https://github.com/alexdoandev/jev-laya-dsh.git
cd dsh-jev-meta

# 1) 权重（一次性）
HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('convaiinnovations/laya-multilingual')"

# 2) ONNX 包（一次性；需 torch）
python3 onnx-export/export_onnx_multilingual.py \
  ~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/<rev> \
  onnx-export/fp32

# 3) 服务
cd onnx-server && npm install
LAYA_PORT=8755 LAYA_MODEL_DIR=../onnx-export/fp32 npm start
curl http://127.0.0.1:8755/health        # {"ok":true,"ready":true,...}
```

### 接入 DeepSeek Harness

向 profile 添加 bundle 与 insert 条目（desktop / web 同理）：

```yaml
# <profile>/package.json → dsh.profile.bundles += "@local/dsh-jev-router"
# <profile>/package.json → dependencies += { "@local/dsh-jev-router": "workspace:*" }
# <profile>/node_modules/@local/dsh-jev-router → 软链到 packages/ 副本
```

```yaml
# <profile>/cordis.patch.yml
- insert:
  - id: local-jev-router
    name: "@local/dsh-jev-router"
    config:
      routing: true
      gate: false
      modelSwap: true
      modelByTier:
        tier2: { provider: zai-coding-cn, model: glm-5.3 }
```

重启应用后每个新回合都会被路由。`packages/dsh-jev-decide`（拉模式工具 `jev_decide`）
与 `/mcp` 端点为可选配套。

## 使用

```bash
node smoke-test.js "<任意 prompt>"        # 端到端路由检查（fake Cordis ctx）
node eval/run-eval.js [--url http://127.0.0.1:8756]   # 61 条评测 + 阈值扫描

curl -X POST http://127.0.0.1:8755/decide \
  -H 'content-type: application/json' \
  -d '{"state":"...","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'

curl http://127.0.0.1:8755/health        # ready / loading / memPressure / idleForSecs
curl -X POST http://127.0.0.1:8755/admin/unload   # 立即释放权重
curl -X POST http://127.0.0.1:8755/admin/load     # 后台预热
```

服务同时支持 MCP（streamable-http，工具 `jev_decide`）：`POST /mcp` —— 任何 MCP
客户端皆可消费该决策层。

## 配置

插件（patch 条目的 `config`）：

| 选项 | 默认 | 说明 |
|---|---|---|
| `routing` | `true` | 在 `agent/pre-step` 上为每个回合分类并注入 notice |
| `gate` | `false` | 启用 `tools/pre-execute` 风险闸门 |
| `gateThreshold` | `0.85` | 触发询问用户的 noul 均值阈值 |
| `gateTools` | `[bash, pwsh]` | 受闸门约束的工具 |
| `timeoutMs` | `2500` | 单请求超时；保证回合不被阻塞 |
| `wake` | `true` | `api-session/added` 时预测唤醒 |
| `wakeUrl` | `http://127.0.0.1:8755/admin/load` | 唤醒端点 |
| `coldRetryMs` | `2000` | 服务唤醒后重试一次，仍失败则 fail-open |
| `maxPromptChars` | `2000` | 作为 Laya state 的 prompt 前缀长度 |
| `modelSwap` | `false` | 经 `agent/request` 将 tier-2 回合固定到 `modelByTier.tier2` |
| `modelByTier` | `{}` | 每层 `{provider, model}` 覆盖 |
| `routeUtterances` | `{}` | 可选：扩展向量路由的种子语句 |

服务端环境变量：`LAYA_PORT`（8755）、`LAYA_MODEL_DIR`、`LAYA_IDLE_UNLOAD_SECS`
（1800）、`LAYA_THREADS`（4）。

## 基准

61 条带金标的越南语 prompt（`eval/eval-set.jsonl`）；阈值扫描由 `eval/run-eval.js`
完成。决策本地运行——无云端 token。

| 信号 | 准确率 |
|---|---|
| 仅 Laya（argmax） | 62.3% |
| 仅向量路由（char-ngram 质心） | 80.3% |
| 仅正则 | 78.7% |
| **融合（Laya × 证据，laya-win ≥ 0.75）** | **86.9%** |

工作点分层：chat 14/20 · 代码 18/20 · 架构 21/21。

### 运行时对比（同一基准、同一问题）

| 指标 | Python/torch | ONNX Runtime (Node) | Rust 宿主 (`ort`) |
|---|---|---|---|
| 冷启动 | 7–20 分钟 | **2.5 秒** | **1.2 秒**（566 KB 二进制） |
| 热态 decide（3 问，闲机） | 0.14–0.36 秒 | **0.29–0.41 秒** | ≤ 0.5 秒（估） |
| 热态 decide（重载窗口） | — | 6.8–13.6 秒 | **1.5–2.4 秒** |
| Parity | 参考实现 | max \|Δlogits\| = 7.4e-06 | 同图同引擎族 |
| 服务磁盘 | venv 922 MB + 1.7 GB 快照 | 1.29 GB 包 + node_modules | 1.29 GB 包 + **566 KB** 二进制 |
| 加载后 RAM | ~0.9–1.8 GB | ~0.9 GB | **~0.57 GB** |

Rust 行（`benchmarks/rust/`）经 `ort`（onnxruntime 的 Rust 绑定）运行同一 ONNX 图，
单一精简二进制——无 Node、无 HTTP 一跳。在重载窗口（三个 `rustc` 任务、load ~80）下，
同一推理比同窗口的 Node 服务**快 4–8×**；空机时两者皆亚秒级，差距退化为 HTTP 一跳。
Rust 是"完全去掉服务进程"嵌入决策层的最强选项。

## 工作原理

- **融合策略** —— Laya 结论与词法/向量证据层合并：一致 ⇒ 采信；分歧 ⇒ 模型置信超过
  调优阈值（0.75）方可推翻证据；低置信 ⇒ 不注入任何 notice。全部阈值位于
  `packages/dsh-jev-router/lib/index.js`，由 `eval/` 扫描选出。
- **向量路由** —— 以 char-trigram + 词级 TF 向量构建每层语句质心，纯 JS、微秒级；
  种子可经 `routeUtterances` 配置。
- **Fail-open** —— Laya 宕机/冷态/超时 ⇒ 本轮跳过路由并触发唤醒；工具闸门同样 fail-open。
- **混合驻留** —— 运行时进程常驻（HTTP 恒在）；权重在 `LAYA_IDLE_UNLOAD_SECS` 空闲或
  macOS 内存压力告警时卸载，加载完成后有 120 秒宽限与在途请求保护。运维详解见
  [docs/OPERATIONS.md](docs/OPERATIONS.md)（容量、杀软扫描风暴、部署清单）。
- **建议而非命令** —— 注入的 notice 只引导智能体，绝不否决回合。唯一能拦截的是闸门
  层，且仅升级为人工确认。

## 项目结构

```
├── laya-server/            Python/torch 参考服务（协议一致）
├── onnx-server/            Node/ONNX 服务（推荐）+ 多语补丁
├── onnx-export/            checkpoint → ONNX 导出脚本
├── packages/
│   └── dsh-jev-router/     DeepSeek Harness 宿主插件（router/gate/swap）
├── eval/                   基准集 + 阈值扫描运行器
├── docs/OPERATIONS.md      运维手册（容量、杀软风暴、部署）
└── smoke-test.js           端到端路由检查
```

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。路由行为变更必须附带 `eval/run-eval.js`
的前后数据。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。第三方组件保留各自许可（Laya 权重：Apache-2.0，
© Convai Innovations —— 本仓库不分发权重；`@receptron/laya`：MIT）。"Jev" 为 TypeSafe
产品；本项目仅使用开源、Jev 兼容的 Laya 模型作为独立本地组件。
