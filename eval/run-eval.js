#!/usr/bin/env node
// run-eval.js — bộ eval tiếng Việt cố định cho jev-router (mục 1/3 của survey).
//
// Nguyên tắc tài nguyên:
// - 0 token GLM: chỉ gọi Laya server local (127.0.0.1:8755), ~0.15s/prompt.
// - 0 RAM resident: chạy xong thoát (launchd khởi động theo lịch rồi tắt).
// - Nếu Laya lạnh: tự wake + chờ tối đa 10 phút rồi mới đo (không bao giờ đốt quota cloud).
//
// Đo: laya-only / vector-only / regex-only / fused(laya × evidence) trên lưới ngưỡng
// laya-win — xuất report JSON + khuyến nghị ngưỡng cho plugin.
//
//   node eval/run-eval.js [--set eval/eval-set.jsonl] [--cold-wait 600]

import { createRequire } from "node:module";
import { execFile } from "node:child_process";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const plugin = require("../packages/dsh-jev-router/lib/index.js");

const setPath = process.argv.includes("--set")
	? process.argv[process.argv.indexOf("--set") + 1]
	: join(here, "eval-set.jsonl");
const BASE = process.argv.includes("--url")
	? process.argv[process.argv.indexOf("--url") + 1].replace(/\/$/, "")
	: "http://127.0.0.1:8755";
const DECIDE = `${BASE}/decide`;
const HEALTH = `${BASE}/health`;
const ADMIN_LOAD = `${BASE}/admin/load`;
const coldWaitS = (() => {
	const i = process.argv.indexOf("--cold-wait");
	return i > -1 ? Number(process.argv[i + 1]) : 600;
})();

const items = readFileSync(setPath, "utf8")
	.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));

function curlJson(url, payload, timeoutS) {
	return new Promise((resolve, reject) => {
		const args = ["-s", "-m", String(timeoutS), url];
		if (payload) args.push("-X", "POST", "-H", "content-type: application/json", "--data-binary", JSON.stringify(payload));
		execFile("/usr/bin/curl", args, { maxBuffer: 1 << 20, timeout: (timeoutS + 2) * 1000 }, (err, stdout) => {
			if (err) return reject(err);
			try { resolve(JSON.parse(stdout)); } catch (e) { reject(e); }
		});
	});
}

async function waitReady() {
	const t0 = Date.now();
	for (;;) {
		try {
			const h = await curlJson(HEALTH, null, 3);
			if (h.ready) return;
			if (!h.loading) await curlJson("http://127.0.0.1:8755/admin/load", {}, 3).catch(() => {});
		} catch { /* server down: keep waiting */ }
		if ((Date.now() - t0) / 1000 > coldWaitS) throw new Error("laya not ready within cold-wait");
		await new Promise((r) => setTimeout(r, 10_000));
	}
}

const centroids = plugin.buildCentroids();
const { fuse, TIER_OF } = plugin;

function layaVerdict(state) { // filled after batch
	void state;
}

const records = [];

await waitReady();
console.log(`eval: ${items.length} prompts — running…`);
const t0 = Date.now();
const ROUTER_QUESTIONS = {
	task_type: {
		type: "choice",
		instructions: "Classify the user request to a coding agent.",
		criteria: {
			chat_qa: "Pure text question, explanation, translation, summary; no code or file changes",
			code_task: "Reading, editing, writing code/files, running commands or tests",
			architecture: "System design, multi-step planning, complex debugging, integration decisions",
		},
	},
};

for (const item of items) {
	const state = "User prompt: " + item.text.slice(0, 2000);
	let laya = { tier: "unknown", p: 0 };
	try {
		let resp = await curlJson(DECIDE, { state, questions: ROUTER_QUESTIONS }, 10);
		if (resp && resp.wakeAccepted) {
			// hybrid-aware: server lạnh → đợi load xong rồi thử lại đúng 1 lần
			await waitReady();
			resp = await curlJson(DECIDE, { state, questions: ROUTER_QUESTIONS }, 10);
		}
		if (resp && resp.ok) {
			const best = plugin.bestChoice(resp.answers.task_type);
			laya = { tier: TIER_OF[best.label] || "unknown", p: best.p };
		}
	} catch { /* record as unknown */ }
	const vec = plugin.vectorTier(item.text, centroids).tier;
	const rex = plugin.regexTier(item.text);
	records.push({ text: item.text, gold: item.gold, laya, vec, rex });
}
const wallS = ((Date.now() - t0) / 1000).toFixed(1);

const acc = (pred) => records.filter((r) => pred(r) === r.gold).length / records.length;
const layaOnly = acc((r) => r.laya.tier);
const vecOnly = acc((r) => r.vec);
const rexOnly = acc((r) => r.rex);

const grid = [];
for (let t = 0.40; t <= 0.90; t += 0.05) {
	const th = Number(t.toFixed(2));
	// reimplements the plugin's fuse() policy with a swept laya-win threshold
	let correct = 0;
	for (const r of records) {
		const modelTier = r.laya.tier;
		const p = r.laya.p;
		const ev = r.vec !== "unknown" ? r.vec : r.rex;
		let tier;
		if (modelTier === "unknown" || p === 0) tier = ev === "unknown" ? "unknown" : ev;
		else if (ev === "unknown") tier = p >= 0.45 ? modelTier : "unknown";
		else if (ev === modelTier) tier = ev;
		else tier = p >= th ? modelTier : ev;
		if (tier === r.gold) correct++;
	}
	grid.push({ layaWinThreshold: th, accuracy: Number((correct / records.length).toFixed(4)) });
}
grid.sort((a, b) => b.accuracy - a.accuracy);
const best = grid[0];

// per-tier confusion for the recommended policy (best threshold)
const fusedTier = (r, th) => {
	const { tier: modelTier, p } = r.laya;
	const ev = r.vec !== "unknown" ? r.vec : r.rex;
	if (modelTier === "unknown" || p === 0) return ev === "unknown" ? "unknown" : ev;
	if (ev === "unknown") return p >= 0.45 ? modelTier : "unknown";
	if (ev === modelTier) return ev;
	return p >= th ? modelTier : ev;
};
const perTier = {};
for (const gold of ["tier1a", "tier1b", "tier2"]) {
	const rows = records.filter((r) => r.gold === gold);
	const ok = rows.filter((r) => fusedTier(r, best.layaWinThreshold) === gold).length;
	perTier[gold] = `${ok}/${rows.length}`;
}

const report = {
	date: new Date().toISOString(),
	set: setPath,
	n: records.length,
	wallSeconds: Number(wallS),
	accuracy: {
		layaOnly: Number(layaOnly.toFixed(4)),
		vectorOnly: Number(vecOnly.toFixed(4)),
		regexOnly: Number(rexOnly.toFixed(4)),
		fusedBest: best,
		grid,
		perTierAtBest: perTier,
	},
	recommendation: best.accuracy >= layaOnly
		? `Giữ fusion; đặt laya-win threshold ≈ ${best.layaWinThreshold} (accuracy ${(best.accuracy * 100).toFixed(1)}%).`
		: `Fusion chưa thắng laya-only (${(layaOnly * 100).toFixed(1)}%) — xem các ca sai trong report trước khi đổi ngưỡng.`,
};

const outDir = join(here, "reports");
mkdirSync(outDir, { recursive: true });
const outFile = join(outDir, `${report.date.slice(0, 10)}.json`);
writeFileSync(outFile, JSON.stringify({ report, records }, null, 2));

console.log(`wall: ${wallS}s · n=${records.length}`);
console.log(`laya-only ${(layaOnly * 100).toFixed(1)}% · vector-only ${(vecOnly * 100).toFixed(1)}% · regex-only ${(rexOnly * 100).toFixed(1)}% · fused(best t=${best.layaWinThreshold}) ${(best.accuracy * 100).toFixed(1)}%`);
console.log(`per-tier (fused): ${JSON.stringify(perTier)}`);
console.log(report.recommendation);
console.log(`report: ${outFile}`);
