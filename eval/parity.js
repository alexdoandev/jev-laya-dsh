// parity.js — Phase A gate: Node/ONNX server vs Rust server must produce
// IDENTICAL answers (4 decimals) on the full VN eval set + probe shapes.
//
// Usage: node eval/parity.js [nodeUrl] [rustUrl]
import { readFileSync } from "node:fs";
import { execFile } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const nodeUrl = process.argv[2] ?? "http://127.0.0.1:8755";
const rustUrl = process.argv[3] ?? "http://127.0.0.1:8757";

const items = readFileSync(join(here, "eval-set.jsonl"), "utf8")
	.trim().split("\n").map((l) => JSON.parse(l));

// the exact router question set (same as the plugin sends)
const QUESTIONS = {
	task_type: {
		type: "choice",
		instructions: "Classify the user request to a coding agent.",
		criteria: {
			chat_qa: "Pure text question, explanation, translation, summary; no code or file changes",
			code_task: "Reading, editing, writing code/files, running commands or tests",
			architecture: "System design, multi-step planning, complex debugging, integration decisions",
		},
	},
	needs_code_model: {
		type: "noul",
		instructions: "Does the request require touching code, files, commands, or tests (as opposed to pure chat)?",
	},
	urgency: {
		type: "score",
		instructions: "How urgent or time-critical is this request?",
		criteria: ["not urgent", "somewhat urgent", "urgent", "critical"],
	},
};

function post(url, payload) {
	return new Promise((resolve, reject) => {
		execFile("/usr/bin/curl", ["-s", "-m", "90", "-X", "POST", url,
			"-H", "content-type: application/json", "--data-binary", JSON.stringify(payload)],
			{ maxBuffer: 1 << 20, timeout: 100000 }, (err, stdout) => {
				if (err) return reject(err);
				try { resolve(JSON.parse(stdout)); } catch (e) { reject(e); }
			});
	});
}

function collect(v, path, out) {
	if (v && typeof v === "object") {
		for (const [k, sub] of Object.entries(v)) collect(sub, `${path}.${k}`, out);
	} else if (typeof v === "number") {
		out.push([path, v]);
	}
}

let mismatches = 0;
let maxDiff = 0;
const checked = items.length;
for (const [i, item] of items.entries()) {
	const payload = { state: "User prompt: " + item.text.slice(0, 2000), questions: QUESTIONS };
	const a = await post(`${nodeUrl}/decide`, payload);
	const b = await post(`${rustUrl}/decide`, payload);
	if (!a.ok || !b.ok) {
		console.log(`[${i}] SERVER ERROR node=${a.ok} rust=${b.ok} — "${item.text.slice(0, 40)}"`);
		mismatches++;
		continue;
	}
	const na = [], nb = [];
	collect(a.answers, "answers", na);
	collect(b.answers, "answers", nb);
	if (na.length !== nb.length) {
		console.log(`[${i}] FIELD COUNT DIFFERS ${na.length} vs ${nb.length}`);
		mismatches++;
		continue;
	}
	for (let j = 0; j < na.length; j++) {
		if (na[j][0] !== nb[j][0]) {
			console.log(`[${i}] PATH DIFFERS: ${na[j][0]} vs ${nb[j][0]}`);
			mismatches++;
			continue;
		}
		const d = Math.abs(na[j][1] - nb[j][1]);
		if (d > maxDiff) maxDiff = d;
		if (d > 1e-4) {
			console.log(`[${i}] DIFF ${na[j][0]}: ${na[j][1]} vs ${nb[j][1]} (Δ=${d.toExponential(2)}) — "${item.text.slice(0, 40)}"`);
			mismatches++;
		}
	}
	void checked;
}
console.log(`\nPARITY GATE: ${checked} prompts · mismatches(>1e-4) = ${mismatches} · max |Δ| = ${maxDiff.toExponential(3)}`);
console.log(mismatches === 0 ? "✅ PARITY PASSED — cut over approved" : "❌ PARITY FAILED — do not cut over");
process.exit(mismatches === 0 ? 0 : 1);
