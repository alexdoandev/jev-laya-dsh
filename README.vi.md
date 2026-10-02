# jev-laya-dsh

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ci](https://github.com/alexdoandev/dsh-jev-meta/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Node](https://img.shields.io/badge/node-%E2%89%A520-brightgreen)
![Runtime](https://img.shields.io/badge/runtime-ONNX%20%7C%20torch-blue)

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | Tiếng Việt
</p>

**jev-laya-dsh** thêm một lớp **quyết định System-1 chạy local** đứng trước các agent
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), theo đúng mô típ
`User → Jev(Laya) → Agent`:

```
User
 │ prompt
 ▼
[1] ROUTER — event agent/pre-step
 │  Laya phân loại prompt (task_type / needs_code_model / urgency) trong một lần
 │  forward; kết quả fusion được inject dạng notice mang tính tham khảo
 ▼
[2] GATE — event tools/pre-execute (tuỳ chọn)
 │  bỏ phiếu 3-probe noul với tool call rủi ro; vượt ngưỡng → hỏi người dùng
 ▼
[3] AGENT — model lớn làm phần việc thật
 │  tuỳ chọn: đổi model theo tier qua event agent/request
 ▼
 câu trả lời
```

Quyết định đến từ **Laya** — model quyết định hiệu chuẩn, mã mở và tương thích Jev
(encoder mmBERT đa ngữ, 100+ ngôn ngữ): không sinh văn bản, không độ tin cậy bịa,
một lần forward, $0/token, 100% local. Model lớn chỉ được gọi cho phần việc thật sự
cần nó.

## Tính năng

- **Router prompt** — mọi prompt được phân loại (chat / code / architecture) với xác suất
  hiệu chuẩn trước khi model chính tốn token đầu tiên; kết luận được inject dạng notice
  của DSH, bản chất là tham khảo.
- **Cổng rủi ro** — bỏ phiếu 3-probe hiệu chuẩn (phá huỷ / không hồi phục / ra ngoài
  workspace) cho lệnh `bash`/`pwsh`, gộp trong một lần forward; vượt ngưỡng thì chuyển
  thành xác nhận của người dùng thay vì chạy âm thầm.
- **Đổi model theo tier** — các turn khó (kiến trúc/lập kế hoạch) có thể ghim vào model
  mạnh cho cả turn qua event `agent/request` (kiểu
  [RouteLLM](https://arxiv.org/abs/2406.18665)/[Hybrid LLM](https://arxiv.org/abs/2404.14618)).
- **Phục vụ hybrid-resident** — runtime ONNX giữ ấm (cold start 2.5 s, quyết định nóng
  0.3 s) trong khi weights tự unload khi rảnh hoặc khi máy thiếu RAM, kèm wake dự đoán
  khi có session mới.
- **Fail-open tuyệt đối** — mọi lớp đều thoái hoá về "không routing" thay vì chặn turn;
  lớp quyết định không bao giờ là điểm chết.
- **0 token cloud cho quyết định** — routing, gate và harness đánh giá chạy 100% local;
  model lớn chỉ tính phí cho công việc thật.

## Yêu cầu

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (bất kỳ bản chạy
  Cordis plugin API nào) cho plugin router — còn các server tự chạy độc lập.
- Node ≥ 20 (server ONNX, eval, smoke test).
- Python 3.12+ với torch — chỉ cần cho lần export ONNX duy nhất.
- Model weights: [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  (Apache-2.0, ~1.3 GB). Nền tảng kiểm chứng: macOS arm64; mọi nơi chạy được
  onnxruntime-node đều đáng kỳ vọng.

## Cài đặt

```bash
git clone https://github.com/alexdoandev/dsh-jev-meta.git
cd dsh-jev-meta

# 1) weights (một lần)
HF_HUB_OFFLINE=0 python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('convaiinnovations/laya-multilingual')"

# 2) ONNX bundle (một lần; cần torch)
python3 onnx-export/export_onnx_multilingual.py \
  ~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots/<rev> \
  onnx-export/fp32

# 3) server
cd onnx-server && npm install
LAYA_PORT=8755 LAYA_MODEL_DIR=../onnx-export/fp32 npm start
curl http://127.0.0.1:8755/health        # {"ok":true,"ready":true,...}
```

### Gắn router vào DeepSeek Harness

Thêm bundle + insert entry vào một profile (giống nhau cho `desktop` và `web`):

```yaml
# <profile>/package.json → dsh.profile.bundles += "@local/dsh-jev-router"
# <profile>/package.json → dependencies += { "@local/dsh-jev-router": "workspace:*" }
# <profile>/node_modules/@local/dsh-jev-router → symlink sang bản trong packages/
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

Khởi động lại app; mọi turn mới sẽ được route. `packages/dsh-jev-decide` (tool
`jev_decide` pull-mode) và endpoint `/mcp` là phần bổ trợ tuỳ chọn.

## Sử dụng

```bash
node smoke-test.js "<prompt bất kỳ>"     # kiểm tra routing end-to-end (fake Cordis ctx)
node eval/run-eval.js [--url http://127.0.0.1:8756]   # benchmark 61 prompt + sweep ngưỡng

curl -X POST http://127.0.0.1:8755/decide \
  -H 'content-type: application/json' \
  -d '{"state":"...","questions":{"urgent":{"type":"noul","instructions":"Is this urgent?"}}}'

curl http://127.0.0.1:8755/health        # ready / loading / memPressure / idleForSecs
curl -X POST http://127.0.0.1:8755/admin/unload   # nhả weights ngay
curl -X POST http://127.0.0.1:8755/admin/load     # khởi động lại nền
```

Server cũng nói MCP (streamable-http, tool `jev_decide`) tại `POST /mcp` — mọi MCP
client đều dùng được lớp quyết định này, không riêng plugin.

## Cấu hình

Plugin (`config` trong patch entry của profile):

| Tuỳ chọn | Mặc định | Mô tả |
|---|---|---|
| `routing` | `true` | phân loại mỗi turn trên `agent/pre-step` và inject notice |
| `gate` | `false` | bật cổng rủi ro trên `tools/pre-execute` |
| `gateThreshold` | `0.85` | ngưỡng mean-noul để escalate hỏi người dùng |
| `gateTools` | `[bash, pwsh]` | tool chịu tác động của gate |
| `timeoutMs` | `2500` | timeout mỗi request; giữ turn không bị khoá |
| `wake` | `true` | dự đoán wake khi có `api-session/added` |
| `wakeUrl` | `http://127.0.0.1:8755/admin/load` | endpoint wake |
| `coldRetryMs` | `2000` | server lạnh: retry một lần rồi fail-open |
| `maxPromptChars` | `2000` | độ dài prompt gửi làm state |
| `modelSwap` | `false` | ghim turn tier-2 sang `modelByTier.tier2` qua `agent/request` |
| `modelByTier` | `{}` | `{provider, model}` cho từng tier |
| `routeUtterances` | `{}` | mở rộng seed câu cho vector router |

Biến môi trường server: `LAYA_PORT` (8755), `LAYA_MODEL_DIR`, `LAYA_IDLE_UNLOAD_SECS`
(1800), `LAYA_THREADS` (4).

## Đánh giá

61 prompt tiếng Việt có nhãn vàng (`eval/eval-set.jsonl`); sweep ngưỡng do
`eval/run-eval.js` thực hiện. Quyết định chạy local — không token cloud.

| Tín hiệu | Độ chính xác |
|---|---|
| Chỉ Laya (argmax) | 62.3% |
| Chỉ vector router (centroid char-ngram) | 80.3% |
| Chỉ regex | 78.7% |
| **Fused (Laya × evidence, laya-win ≥ 0.75)** | **86.9%** |

Theo tier tại điểm vận hành: chat 14/20 · code 18/20 · architecture 21/21.

### So sánh runtime (cùng benchmark, cùng câu hỏi)

| Chỉ số | Python/torch | ONNX Runtime (Node) | Rust host (`ort`) |
|---|---|---|---|
| Cold start | 7–20 phút | **2.5 giây** | **1.2 giây** (nhị phân 566 KB) |
| Decide nóng (3 câu, máy rảnh) | 0.14–0.36 giây | **0.29–0.41 giây** | ≤ 0.5 giây (ước tính) |
| Decide nóng (cửa sổ CPU quá tải) | — | 6.8–13.6 giây | **1.5–2.4 giây** |
| Parity | tham chiếu | max \|Δlogits\| = 7.4e-06 | cùng graph, cùng họ engine |
| Disk phục vụ | venv 922 MB + snapshot 1.7 GB | bundle 1.29 GB + node_modules | bundle 1.29 GB + **nhị phân 566 KB** |
| RAM (đã load) | ~0.9–1.8 GB | ~0.9 GB | **~0.57 GB** |

Dòng Rust (`benchmarks/rust/`) chạy đúng graph ONNX qua `ort` (binding onnxruntime cho
Rust) bằng một nhị phân gọn đơn nhất — không Node, không bước HTTP. Trong cửa sổ máy
quá tải (3 tiến trình `rustc`, load ~80), cùng một phép suy luận **nhanh hơn 4–8× so
với Node service trong cùng cửa sổ**; máy rảnh thì cả hai đều dưới 0.5 giây và chênh
lệch thu về đúng một bước HTTP. Rust là lựa chọn mạnh nhất nếu muốn nhúng lớp quyết
định không cần server process.

## Cách hoạt động

- **Chính sách fusion** — verdict của Laya và evidence tier (lớp từ vựng/regex) được kết
  hợp: trùng nhau ⇒ chấp nhận; khác nhau ⇒ model phải vượt ngưỡng tin cậy đã tinh chỉnh
  (0.75) mới lật được evidence; độ tin cậy thấp ⇒ không notice. Toàn bộ ngưỡng nằm trong
  `packages/dsh-jev-router/lib/index.js`, được chọn bằng sweep trong `eval/`.
- **Vector router** — centroid câu nói theo vector char-trigram + word TF, thuần JS,
  micro giây; seed chỉnh được qua `routeUtterances`.
- **Fail-open** — Laya chết, lạnh hoặc chậm ⇒ bỏ routing cho turn đó và kích wake;
  gate cũng fail-open y hệt.
- **Hybrid residency** — process runtime nằm thường trú (HTTP luôn lắng nghe); weights
  unload sau `LAYA_IDLE_UNLOAD_SECS` rảnh hoặc khi macOS báo memory pressure, có grace
  120 giây sau load và bảo vệ request đang chạy. Chi tiết vận hành xem
  [docs/OPERATIONS.md](docs/OPERATIONS.md).
- **Tham khảo chứ không mệnh lệnh** — notice chỉ dẫn hướng agent; không bao giờ phủ quyết
  turn. Thành phần duy nhất có thể chặn là gate, và nó chỉ escalate lên con người.

## Cấu trúc project

```
├── laya-server/            server tham chiếu Python/torch (cùng protocol)
├── onnx-server/            server Node/ONNX (khuyến nghị) + patch đa ngữ
├── onnx-export/            script export checkpoint → ONNX
├── packages/
│   └── dsh-jev-router/     plugin host DeepSeek Harness (router/gate/swap)
├── eval/                   benchmark set + runner sweep ngưỡng
├── docs/OPERATIONS.md      sổ tay vận hành (sizing, bão AV, deploy)
└── smoke-test.js           kiểm tra routing end-to-end
```

## Đóng góp

Xem [CONTRIBUTING.md](CONTRIBUTING.md). Thay đổi hành vi routing phải kèm số liệu
trước/sau từ `eval/run-eval.js`.

## Giấy phép

MIT — xem [LICENSE](LICENSE). Các thành phần third-party giữ giấy phép riêng
(weights Laya: Apache-2.0, © Convai Innovations — không phân phối lại weights trong
repo; `@receptron/laya`: MIT). "Jev" là sản phẩm của TypeSafe; project này chỉ dùng
Laya — bản mở, tương thích Jev — như một thành phần độc lập chạy local.
