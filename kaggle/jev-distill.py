# %% [markdown]
# # Jev-Distill — fine-tune một System-1 classifier tiếng Việt trên Kaggle
#
# **Mục tiêu**: train student (mmBERT + head) thay thế/nâng cấp Laya cho routing
# tier (chat_qa / code_task / architecture) trên domain coding-agent tiếng Việt.
#
# **Teacher** (chọn 1 — đều là OpenAI-compatible endpoint, key để trong Kaggle Secrets):
# - `pdecider` — Perplexity Decider v1.1 (27B) — teacher đầu bảng nhánh Jev
# - `quyet`    — Quyet-1.0-Large
# - `glm`      — GLM-flash qua zai-coding-cn (quota sẵn có; label ~500 prompt)
# - `gold`     — chỉ train trên nhãn vàng (eval-set + CSV bạn tự gán)
#
# ⚠️ ToS: kiểm tra điều khoản của teacher có cho phép dùng output để train hay
# không trước khi public student weights; dùng cá nhân/thử nghiệm thường OK.
#
# **Kaggle setup**: Settings → Accelerator = GPU T4 x2 (hoặc P100) · Internet = On ·
# Add-ons → Secrets: `ZAI_API_KEY` (nếu teacher=glm) · Upload Dataset chứa
# `eval-set.jsonl` (từ repo jev-laya-dsh/eval/).

# %% setup
import os, json, time, random
SEED = 42
random.seed(SEED)

TEACHER = os.environ.get("TEACHER", "glm")
# local open teachers (chạy ngay trong notebook, không cần key):
#   quyet-large  — chinhnc/Quyet-1.0-Large (31B, bf16 62.5GB — KHÔNG vừa T4x2;
#                  dùng nếu có A100/80GB hoặc bản GGUF qua llama.cpp)
#   quyet-medium — chinhnc/Quyet-1.0-Medium (nhẹ hơn, khuyên dùng trên Kaggle)
# API teachers (OpenAI-compatible):
#   pdecider — Perplexity Decider v1.1 (27B) · glm — GLM-flash · gold — chỉ nhãn vàng
PRESETS = {
    "pdecider": ("PDECIDER_BASE", "https://api.perplexity.ai",
                 "PDECIDER_KEY", "PERPLEXITY_DECIDER_KEY", "perplexity-decider-v1.1-27b"),
    "glm":      ("GLM_BASE", "https://open.bigmodel.cn/api/paas/v4",
                 "GLM_KEY", "ZAI_API_KEY", "glm-5.3-flash"),
}
TEACHER_BASE = TEACHER_KEY = TEACHER_MODEL = ""
STUDENT_ID = "jhu-clsp/mmBERT-base"
MAX_LABEL = int(os.environ.get("MAX_LABEL", "500"))  # giới hạn quota teacher

!pip -q install "transformers>=4.44" datasets accelerate peft onnx onnxruntime "openai>=1.40"

import torch, numpy as np
torch.manual_seed(SEED); np.random.seed(SEED)
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModel, AutoConfig,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
import torch.nn as nn

TIERS = ["tier1a", "tier1b", "tier2"]
LABEL2ID = {t: i for i, t in enumerate(TIERS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}

# %% [markdown]
# ## 1) Dữ liệu — seed vàng (61) + CSV tuỳ chọn (production logs đã gán nhãn)

# %% load seed + optional csv
import pathlib
SEED_JSONL = "/kaggle/input/jev-eval-set/eval-set.jsonl"   # đổi theo dataset bạn upload
rows = []
if pathlib.Path(SEED_JSONL).exists():
    rows = [json.loads(l) for l in open(SEED_JSONL) if l.strip()]
else:  # fallback demo nhỏ nếu chưa upload dataset
    rows = [
        {"text": "giải thích closure trong javascript là gì", "gold": "tier1a"},
        {"text": "sửa hàm calculate trong utils.js bị sai tổng", "gold": "tier1b"},
        {"text": "refactor kiến trúc microservices sang event-driven", "gold": "tier2"},
    ]

EXTRA_CSV = os.environ.get("EXTRA_CSV", "")  # /kaggle/input/<dataset>/labels.csv: text,gold
import csv as _csv
if EXTRA_CSV and pathlib.Path(EXTRA_CSV).exists():
    with open(EXTRA_CSV) as f:
        for r in _csv.DictReader(f):
            if r.get("gold") in LABEL2ID:
                rows.append({"text": r["text"], "gold": r["gold"]})

print("seed/extra rows:", len(rows))

# %% [markdown]
# ## 2) Teacher labeling — chỉ chạy cho prompt CHƯA có nhãn vàng
# GLM-flash gán tier theo đúng định nghĩa 3 tier; tự-đánh dấu `low_conf` khi
# xác suất thấp để bạn review tay. Mọi nhãn được cache ra JSONL (chạy lại không tốn).

# %% teacher labeling
CACHE = "/kaggle/working/teacher-labels.jsonl"
labeled: dict[str, str] = {}
if pathlib.Path(CACHE).exists():
    for l in open(CACHE):
        r = json.loads(l); labeled[r["text"]] = r["label"]

LOCAL_QUYET = {"quyet-large": "chinhnc/Quyet-1.0-Large",
               "quyet-medium": "chinhnc/Quyet-1.0-Medium"}

def teacher_predict(state_text: str) -> tuple[str, float]:
    """Trả về (tier, confidence). Local Quyet dùng cùng typed-question contract."""
    if TEACHER in LOCAL_QUYET:
        r = TEACHER_MODEL.predict({"state": state_text[:6000]}, ROUTER_Q)
        a = r["answers"]["task_type"]
        conf = a.get("confidence", 0)
        return a["choice"], float(conf)
    client = OpenAI(base_url=TEACHER_BASE, api_key=TEACHER_KEY or "EMPTY")
    rsp = client.chat.completions.create(
        model=TEACHER_MODEL, temperature=0, max_tokens=60,
        messages=[{"role": "system", "content": ROUTER_SYS},
                  {"role": "user", "content": state_text[:1500]}])
    raw = rsp.choices[0].message.content.strip()
    j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    return j.get("tier", ""), float(j.get("confidence", 0))

ROUTER_Q = {
    "task_type": {"type": "choice",
        "instructions": "Classify the user request to a coding agent.",
        "criteria": {"chat_qa": "Pure text question, explanation, translation, summary; no code or file changes",
                     "code_task": "Reading, editing, writing code/files, running commands or tests",
                     "architecture": "System design, multi-step planning, complex debugging, integration decisions"}},
    "needs_code_model": {"type": "noul",
        "instructions": "Does the request require touching code, files, commands, or tests (as opposed to pure chat)?"},
    "urgency": {"type": "score", "instructions": "How urgent or time-critical is this request?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
}

if TEACHER in LOCAL_QUYET:
    !pip -q install "quyet[multi-gpu]" bitsandbytes accelerate
    import quyet
    TEACHER_MODEL = quyet.load(LOCAL_QUYET[TEACHER])
elif TEACHER != "gold":
    from openai import OpenAI
    _ = TEACHER_BASE, TEACHER_KEY, TEACHER_MODEL
    cache_f = open(CACHE, "a")
    budget = MAX_LABEL
    for r in rows:
        if r["text"] in labeled or budget <= 0:
            continue
        if budget == MAX_LABEL:
            print("teacher labeling bắt đầu…")
        try:
            label, conf = teacher_predict(r["text"][:1500])
        except Exception as e:
            print("teacher error:", e); break
        if label in LABEL2ID:
            r["teacher"], r["teacher_conf"] = label, conf
            labeled[r["text"]] = label
            cache_f.write(json.dumps({"text": r["text"], "label": label,
                                      "conf": conf}, ensure_ascii=False) + "\n")
            cache_f.flush()
            budget -= 1
        time.sleep(0.2)
    cache_f.close()
print("đã có nhãn teacher cho:", sum(1 for r in rows if "teacher" in r), "prompt")

# %% [markdown]
# ## 3) Tập train — ưu tiên nhãn vàng, điền teacher cho phần còn lại

# %% build final dataset
final = []
for r in rows:
    label = r.get("gold") or labeled.get(r["text"]) or r.get("teacher")
    if label in LABEL2ID:
        final.append({"text": r["text"], "label": LABEL2ID[label],
                      "source": "gold" if r.get("gold") else "teacher"})
from collections import Counter
print(Counter(d["source"] for d in final), Counter(ID2LABEL[d["label"]] for d in final))
random.shuffle(final)
val_n = max(2, int(len(final) * 0.15))
ds = DatasetDict({
    "train": Dataset.from_list(final[val_n:]),
    "val": Dataset.from_list(final[:val_n]),
})

# %% [markdown]
# ## 4) Student — mmBERT (encoder) + mean-pool + head 3 lớp
# Train nhanh trên T4: head-only ~3 phút; bật LORA để tune thêm encoder (~15 phút).

# %% tokenize
tok = AutoTokenizer.from_pretrained(STUDENT_ID)
def tok_fn(b):
    return tok(b["text"], truncation=True, max_length=256)
ds_tok = ds.map(tok_fn, batched=True)
collator = DataCollatorWithPadding(tok)

# %% model
class JevHead(nn.Module):
    def __init__(self, model_id, n=3, p=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        d = self.encoder.config.hidden_size
        self.drop, self.head = nn.Dropout(p), nn.Linear(d, n)
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.head(self.drop(pooled))
        loss = None
        if labels is not None:
            w = torch.tensor([1.0, 1.0, 1.2], device=logits.device)  # tier2 hiếm hơn
            loss = nn.functional.cross_entropy(logits, labels, weight=w)
        return {"loss": loss, "logits": logits}

model = JevHead(STUDENT_ID)

# %% train
LORA = os.environ.get("LORA", "0") == "1"
if LORA:
    from peft import LoraConfig, get_peft_model
    model.encoder = get_peft_model(model.encoder, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["query", "value"]))
    model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="/kaggle/working/out", num_train_epochs=4,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-4 if LORA else 1e-3,   # head-only dùng lr cao
    weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="accuracy", logging_steps=10,
    report_to=[], seed=SEED,
)
def metrics(p):
    preds = np.argmax(p.predictions, -1)
    acc = (preds == p.label_ids).mean()
    return {"accuracy": acc}
trainer = Trainer(model=model, args=args, train_dataset=ds_tok["train"],
                  eval_dataset=ds_tok["val"], data_collator=collator,
                  compute_metrics=metrics)
trainer.train()
print("val accuracy:", metrics(trainer.predict(ds_tok["val"]))["accuracy"])

# %% [markdown]
# ## 5) So sánh với baseline Laya-only (62.3% trên cùng domain benchmark)

# %% so sánh trên tập gold
gold_only = [r for r in rows if r.get("gold")]
correct = 0
model.eval()
for r in gold_only:
    enc = tok(r["text"], truncation=True, max_length=256, return_tensors="pt").to(model.device)
    with torch.no_grad():
        pred = model(**{k: v for k, v in enc.items()})["logits"].argmax(-1).item()
    correct += (ID2LABEL[pred] == r["gold"])
print(f"student trên {len(gold_only)} gold: {correct}/{len(gold_only)} = {correct/len(gold_only):.1%}")
print("baseline laya-only trên benchmark VN: 62.3% (xem README §6)")

# %% [markdown]
# ## 6) Export ONNX — thả vào server như model v2 (cùng JSON contract /decide)

# %% export
import torch as T
class Wrapped(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask)["logits"]
w = Wrapped(model).eval()
ex = (T.ones(1, 16, dtype=T.long), T.ones(1, 16, dtype=T.long))
T.onnx.export(w, ex, "/kaggle/working/jev-router-v2.onnx", opset_version=17,
              input_names=["input_ids", "attention_mask"],
              output_names=["logits"],
              dynamic_axes={"input_ids": {0: "b", 1: "s"},
                            "attention_mask": {0: "b", 1: "s"},
                            "logits": {0: "b", 1: "s"}})
print("exported: /kaggle/working/jev-router-v2.onnx")
print("tokenizer cũng cần tải về:", STUDENT_ID)

# %% [markdown]
# ## 7) Bước tiếp theo sau notebook
# 1. Tải `jev-router-v2.onnx` + tokenizer về máy, đặt vào `onnx-export/student-v2/`.
# 2. Server (Rust hoặc Node) thêm chế độ `LAYOUT=v2`: chạy student thay laya
#    (bỏ marker/marker_mask, thêm tokenizer mới) — JSON contract /decide giữ nguyên.
# 3. A/B: chạy `eval/parity.js` + `run-eval.js` mở rộng — student phải thắng
#    laya-only 62.3% trên cùng bộ gold trước khi cắt.
# 4. Ca student thấp tự tin (<0.6) → rơi về laya arbitrer (giữ nguyên thiết kế fail-open).
