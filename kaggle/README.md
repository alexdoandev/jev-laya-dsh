# Jev-Distill trên Kaggle — teacher đầu bảng nhánh Jev

Notebook: **`jev-distill.py`** — distill một System-1 classifier tiếng Việt
(mmBERT-base + head 3 lớp) từ teacher đầu bảng nhánh Jev, chạy **miễn phí trên
Kaggle GPU**, xuất ONNX drop-in cho server.

## 1. Chuẩn bị (10 phút)

1. Tài khoản Kaggle → **Create → New Notebook**.
2. **File → Import Notebook** → upload `jev-distill.py` (hoặc dán vào 1 code cell —
   các dòng `!pip` là magic, chạy bình thường trên Kaggle).
3. **Settings (panel phải)**:
   - Accelerator: **GPU T4 x2** (hoặc P100)
   - Internet: **On**
4. **Add-ons → Secrets** — thêm các secret teacher bạn có:

| Secret | Dùng cho | Lấy ở đâu |
|---|---|---|
| `PDECIDER_KEY` | Perplexity Decider v1.1 (27B) | trang API của Perplexity |
| `QUYET_KEY` | Quyet-1.0-Large | trang/leaderboard của Quyet |
| `ZAI_API_KEY` | GLM-flash (fallback free-quota) | key zai-coding-cn bạn đang có |

5. **Upload Dataset** chứa `eval/eval-set.jsonl` từ repo (61 prompt gold) — hoặc
   để notebook chạy chế độ demo 3 prompt nếu chưa upload.

## 2. Chạy

| Bước | Cell | Thời gian (T4) | Ghi chú |
|---|---|---|---|
| 1 | Setup + installs | ~2 phút | |
| 2 | Load dữ liệu | < 1 phút | 61 gold (+ CSV production nếu có) |
| 3 | **Teacher labeling** | phụ thuộc teacher | `MAX_LABEL=500` mặc định; cache ra `teacher-labels.jsonl` nên chạy lại không tốn |
| 4 | Train student | **10–20 phút** | mmBERT-base + head; `LORA=1` nếu muốn tune thêm encoder |
| 5 | Eval vs laya-only | < 1 phút | baseline cần vượt: 62.3% |
| 6 | Export ONNX | < 1 phút | `jev-router-v2.onnx` + tokenizer |

Output nằm ở **/kaggle/working/**: `jev-router-v2.onnx`, `out/`, `teacher-labels.jsonl`.

## 3. Chọn teacher

Đặt ở cell config: `TEACHER = "pdecider"` | `"quyet"` | `"glm"` | `"gold"`.

- **Perplexity Decider v1.1 (27B)** / **Quyet-1.0-Large**: teacher đầu bảng nhánh
  Jev — chất lượng nhãn cao nhất; điền thêm base URL nếu endpoint khác mặc định
  (`PDECIDER_BASE`, `QUYET_BASE`).
- **GLM-flash**: fallback không tốn thêm tiền (quota sẵn có), khuyên label ~500
  prompt đầu để ước lượng chất lượng trước khi mở rộng.
- **`gold`**: bỏ qua teacher — chỉ dùng nhãn vàng.

⚠️ **ToS**: kiểm tra điều khoản teacher (Perplexity/Quyet) về việc dùng output để
train model — dùng cá nhân/thử nghiệm thường OK; **public student weights** có thể
vi phạm ToS của teacher proprietary — cân nhắc trước khi mở source student.

## 4. Đưa về production

1. Tải `jev-router-v2.onnx` + tokenizer (từ `jhu-clsp/mmBERT-base`) về máy.
2. Đặt vào `onnx-export/student-v2/`.
3. Server (Rust hoặc Node) thêm chế độ đọc student — JSON contract `/decide`
   giữ nguyên; turn thấp tự tin (<0.6) rơi về laya arbitrer như thiết kế fail-open.
4. Gate chấp nhận: student ≥ 86.9% (fused baseline) trên eval-set trước khi cắt.

## 5. Chi phí

- Kaggle GPU: **30 giờ/tuần miễn phí** — notebook này dùng ~30–45 phút GPU.
- Teacher API: tuỳ model (Decider/Quyet có thể tính per-call — label 500 prompt
  ~200k token input; xem giá trang teacher). GLM-flash đi theo quota sẵn có.
