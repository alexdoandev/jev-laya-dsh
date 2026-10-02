"""Export the Laya multilingual checkpoint to ONNX — adapted from
receptron/laya export/export_onnx.py: build_model comes from the installed
`laya` package (laya.common) instead of the English repo's rl_common.py.

Usage: ~/.venvs/laya/bin/python export_onnx_multilingual.py [model_dir] [out_dir]
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

from laya.common import build_model  # noqa: E402

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual/"
    "snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
)
model_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else SNAP)
out_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "fp32-legacy")
os.makedirs(out_dir, exist_ok=True)

from safetensors.torch import load_file  # noqa: E402

cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
model.eval()
model.encoder.config.reference_compile = False


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, torch.softmax(act.float(), -1)


w = Wrapper(model)
B, L, K = 2, 40, 4
ex = (
    torch.randint(5, 1000, (B, L)),
    torch.ones(B, L, dtype=torch.long),
    torch.tensor([[3, 9, 15, 21], [3, 9, 0, 0]]),
    torch.tensor([[True, True, True, True], [True, True, False, False]]),
    torch.tensor([0, 2]),
)
ex[1][1, 30:] = 0  # second row padded

out = os.path.join(out_dir, "laya.onnx")
# grad must stay enabled: nn.TransformerEncoderLayer's fused "fast path" (not
# exportable) is only taken under no_grad — keep parity with upstream script.
torch.onnx.export(
    w, ex, out, opset_version=18, do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
    output_names=["logits", "act_probs"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "marker_pos": {0: "batch", 1: "options"}, "marker_mask": {0: "batch", 1: "options"},
                  "qtype": {0: "batch"}},
)

shutil.copytree(os.path.join(model_dir, "tokenizer"), os.path.join(out_dir, "tokenizer"), dirs_exist_ok=True)
json.dump({k: cfg[k] for k in ("max_len", "head_max_len", "temperature", "temperature_by_options")},
          open(os.path.join(out_dir, "laya_config.json"), "w"), indent=1)

# parity check vs torch (same tensors as export example)
import onnxruntime as ort  # noqa: E402

with torch.no_grad():
    ref_logits, ref_act = w(*ex)
sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
o = sess.run(None, {"input_ids": ex[0].numpy(), "attention_mask": ex[1].numpy(), "marker_pos": ex[2].numpy(),
                    "marker_mask": ex[3].numpy(), "qtype": ex[4].numpy()})
print("max |dlogits| =", np.abs(o[0] - ref_logits.numpy()).max(), " max |dact| =", np.abs(o[1] - ref_act.numpy()).max())
print("wrote", out, "%.0f MB" % (sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.startswith("laya")) / 1e6))
