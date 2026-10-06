// @local/dsh-jev-router — "User -> Jev (Laya) -> Agent" for DeepSeek Harness.
//
// Layer 1 (this plugin): every user prompt is classified by the warm local Laya
// System One decision server (launchd, 127.0.0.1:8755) BEFORE the big model
// spends its first token. The verdict is appended to the step as an advisory
// notice message, so the agent sees routing evidence without any schema change.
// Optional layer 2: a noul risk gate on tool calls (ask-the-human escalation).
//
// Design rules:
// - Fail-open. Laya dead/timeout/parse error => the turn runs untouched.
// - Route once per turn (first step that carries a user message).
// - Never mutates user content; only appends a source-branded notice.

const URL_DEFAULT = "http://127.0.0.1:8755/decide";
const WAKE_URL_DEFAULT = "http://127.0.0.1:8755/admin/load";
const CURL = "/usr/bin/curl";

const name = "jev-router";
const inject = ["subprocess", "systemPrompt", "tools"];

// ---------- tiny helpers ----------

function postDecide(subprocess, url, payload, timeoutS) {
	return new Promise((resolve, reject) => {
		let handle;
		try {
			handle = subprocess.spawn({
				argv: [CURL, "-s", "-m", String(timeoutS), "-X", "POST", url,
					"-H", "content-type: application/json",
					"--data-binary", JSON.stringify(payload)],
				cwd: "/tmp",
				stdio: { stdin: "ignore", stdout: { maxBytes: 1 << 20 }, stderr: { maxBytes: 1 << 16 } },
				graceMs: 1500,
			});
		} catch (error) {
			return reject(error);
		}
		handle.done.then(() => {
			try {
				const r = handle.collected.stdout.readFrom(0);
				const text = r && r.text ? r.text.trim() : "";
				if (!text) return reject(new Error("empty response from laya-warm"));
				resolve(JSON.parse(text));
			} catch (error) {
				reject(error);
			}
		}).catch(reject);
	});
}

function notice(text, summary) {
	return Object.freeze(structuredClone({
		id: randomUuid(),
		role: "user",
		content: [{ type: "text", text }],
		source: { kind: "jev-router", form: "notice", summary },
	}));
}

function randomUuid() {
	const g = globalThis.crypto;
	if (g && typeof g.randomUUID === "function") return g.randomUUID();
	// RFC4122 v4 fallback (host node >= 20 always has randomUUID; belt and braces).
	return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
		const r = (Math.random() * 16) | 0;
		const v = c === "x" ? r : (r & 0x3) | 0x8;
		return v.toString(16);
	});
}

// Fire-and-forget wake of the hybrid server (weights-only reload, ~seconds).
function wake(subprocess, url) {
	try {
		const handle = subprocess.spawn({
			argv: [CURL, "-s", "-m", "2", "-X", "POST", url],
			cwd: "/tmp",
			stdio: { stdin: "ignore", stdout: { maxBytes: 4096 }, stderr: { maxBytes: 4096 } },
			graceMs: 500,
		});
		handle.done.catch(() => {});
	} catch {
		// never let a wake failure surface
	}
}

function bestChoice(answer) {
	if (!answer || !answer.probabilities) return { label: "unknown", p: 0 };
	let label = "unknown";
	let p = 0;
	for (const [k, v] of Object.entries(answer.probabilities)) {
		if (typeof v === "number" && v > p) {
			p = v;
			label = k;
		}
	}
	return { label, p };
}

function clamp01(v) {
	return typeof v === "number" && v >= 0 && v <= 1 ? v : undefined;
}

// ---------- questions asked of Laya (state format won the 24/09 A/B: raw prompt, no invented features) ----------

function routerQuestions() {
	return {
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
}

function gateQuestion(toolName) {
	// Tabi-style tiny ensemble: 3 typed probes batched in ONE forward pass
	// (same ~latency as a single question). Ask-user if mean is high OR any
	// single probe is near-certain.
	return {
		destructive: {
			type: "noul",
			instructions: `Is this ${toolName} tool call destructive — does it delete, overwrite or damage data?`,
		},
		irreversible: {
			type: "noul",
			instructions: `Is this ${toolName} tool call hard to reverse or undo afterwards?`,
		},
		outside_workspace: {
			type: "noul",
			instructions: `Does this ${toolName} tool call reach outside the current workspace or affect the user's system globally?`,
		},
	};
}

function gateVote(answers) {
	const vals = ["destructive", "irreversible", "outside_workspace"]
		.map((k) => clamp01((answers || {})[k] && answers[k].noul))
		.filter((v) => v !== undefined);
	if (!vals.length) return undefined;
	const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
	const max = Math.max(...vals);
	return { mean, max, hits: vals.filter((v) => v >= 0.8).length };
}

const TIER_OF = { chat_qa: "tier1a", code_task: "tier1b", architecture: "tier2" };

// Regex evidence — port of the proven laya-route.sh tiers (ZCode A/B 24/09: 4/4).
const RE_TIER2 = /(kiến trúc|thiết kế|architecture|design|plan\b|kế hoạch|strategy|refactor (to|lớn|major)|cross-?service|security|bảo mật|tối ưu hiệu năng|performance|race condition|leak|deadlock|integrate|tích hợp|hệ thống|tradeoff|so sánh.*giải pháp)/i;
const RE_TIER1B = /(code|file|function|func|hàm|biến|variable|class|module|component|refactor|lint|build|compile|chạy|run|test|spec|commit|git |branch|merge|PR\b|api|endpoint|sql|query|config|import|export|syntax|typo|đổi tên|rename|format|sửa|đọc|read|xem.*file|mở.*file|grep|find|tìm.*file|viết.*code|thêm.*function|fix.*bug|bug|lỗi|patch)/i;
const RE_TIER1A = /(giải thích|explain|dịch|translate|tóm tắt|summary|summarize|khái niệm|concept|so sánh.*khái niệm| khác biệt giữa|định nghĩa|definition|hướng dẫn|gợi ý|ý tưởng|brainstorm|câu hỏi|là gì|what is)/i;

function regexTier(text) {
	if (RE_TIER2.test(text)) return "tier2";
	if (RE_TIER1B.test(text)) return "tier1b";
	if (RE_TIER1A.test(text)) return "tier1a";
	if (text.length < 120) return "tier1a";
	return "unknown";
}

// ---------- vector router (Semantic-Router-style centroids, sparse lexical vectors) ----------
// Char-trigram + word TF vectors compared against per-tier utterance centroids.
// Pure JS: a few KB of memory, microseconds per prompt, works when Laya is cold.
// No new resident model — the hardware-frugal alternative to dense embeddings.

const DEFAULT_UTTERANCES = {
	tier1a: [
		"giải thích khái niệm này cho tôi", "offset trong sql là gì", "giải thích closure là gì",
		"dịch đoạn văn này sang tiếng anh", "tóm tắt bài viết sau", "khác biệt giữa rest và graphql là gì",
		"so sánh docker và virtualmachine", "hướng dẫn cách học git", "cho tôi ý tưởng đặt tên biến",
		"what is a deadlock", "giải thích big-o notation", "tóm tắt nội dung file README này",
		"câu hỏi về design pattern", "giải thích cách hoạt động của https",
	],
	tier1b: [
		"sửa giúp hàm calculate trong utils.js", "fix bug login không redirect", "thêm function validate email",
		"chạy test rồi báo kết quả", "refactor hàm này cho gọn", "đổi tên biến code cho dễ đọc",
		"mở file config xem giúp", "commit các thay đổi vừa rồi", "viết unit test cho hàm parse",
		"format lại file theo prettier", "grep tìm tất cả chỗ gọi api này", "thêm arg CLI cho script",
		"cập nhật dependency trong package.json", "xử lý lỗi undefined khi parse json",
	],
	tier2: [
		"refactor kiến trúc microservices sang event-driven", "thiết kế hệ thống thanh toán chịu tải cao",
		"lập kế hoạch migration database từng bước", "phân tích tradeoff giữa các giải pháp queue",
		"review bảo mật toàn bộ api gateway", "debug race condition phức tạp nhiều service",
		"thiết kế schema cho đa năng lượng", "plan tích hợp hệ thống crm với erp",
		"đánh giá kiến trúc monolith có nên tách không", "chiến lược scale từ 1k lên 100k user",
		"thiết kế caching layer cho toàn hệ thống", "kế hoạch refactor cross-service",
	],
};

function normalizeForVec(text) {
	return String(text).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

function textToVec(text) {
	const s = normalizeForVec(text);
	const vec = new Map();
	const bump = (k) => vec.set(k, (vec.get(k) || 0) + 1);
	const words = s.split(" ").filter(Boolean);
	for (const w of words) if (w.length >= 2) bump("w:" + w);
	for (const w of words) {
		if (w.length < 3) continue;
		for (let i = 0; i + 3 <= w.length; i++) bump("g:" + w.slice(i, i + 3));
	}
	return vec;
}

function cosine(a, b) {
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (const [, v] of a) na += v * v;
	for (const [k, v] of b) {
		nb += v * v;
		const av = a.get(k);
		if (av !== undefined) dot += av * v;
	}
	if (!na || !nb) return 0;
	return dot / Math.sqrt(na * nb);
}

function buildCentroids(overrides) {
	const merged = {};
	for (const tier of Object.keys(DEFAULT_UTTERANCES)) {
		merged[tier] = [...DEFAULT_UTTERANCES[tier], ...((overrides && overrides[tier]) || [])];
	}
	const centroids = {};
	for (const [tier, utts] of Object.entries(merged)) {
		const acc = new Map();
		for (const u of utts) {
			for (const [k, v] of textToVec(u)) acc.set(k, (acc.get(k) || 0) + v);
		}
		centroids[tier] = acc;
	}
	return centroids;
}

// returns { tier, margin, top } — margin = top1 - top2 cosine
function vectorTier(text, centroids) {
	const v = textToVec(text);
	let best = "unknown";
	let top = -1;
	let second = -1;
	for (const [tier, c] of Object.entries(centroids)) {
		const s = cosine(v, c);
		if (s > top) {
			second = top;
			top = s;
			best = tier;
		} else if (s > second) second = s;
	}
	return { tier: top >= 0.1 ? best : "unknown", margin: top - second, top };
}

// Combined evidence: vector verdict wins when confident; regex covers the tail.
function evidenceTier(text, centroids) {
	const v = vectorTier(text, centroids);
	const r = regexTier(text);
	if (v.tier !== "unknown") return v.margin >= 0.02 || r === "unknown" || r === v.tier ? v.tier : r;
	return r;
}

// Fusion (thresholds tuned by eval/run-eval.js on 61 VN prompts, 02/10:
// laya-win @0.75 → 86.9% fused accuracy; see eval/reports/).
function fuse(modelTier, modelP, rTier) {
	if (modelTier === "unknown" || modelP === 0) return rTier === "unknown" ? "unknown" : { tier: rTier, src: "regex-only" };
	if (rTier === "unknown") return modelP >= 0.45 ? { tier: modelTier, src: "laya-solo" } : { tier: "unknown", src: "low-conf" };
	if (rTier === modelTier) return { tier: rTier, src: "agree" };
	return modelP >= 0.75 ? { tier: modelTier, src: "laya-win" } : { tier: rTier, src: "regex-win" };
}

function buildHint(tier, verdict, config) {
	if (tier === "unknown") return "";
	const lines = [];
	const model = config.modelByTier && config.modelByTier[tier];
	if (tier === "tier1a") {
		lines.push("[Routing|jev] Tier 1a (pure chat/Q&A) — answer directly and keep it light; no heavy agents/tools needed.");
	} else if (tier === "tier1b") {
		lines.push("[Routing|jev] Tier 1b (routine code) — work directly in the workspace; do not inflate the scope.");
	} else {
		lines.push("[Routing|jev] Tier 2 (architecture/planning/hard debugging) — plan step-by-step before editing; delegate mechanical parts when possible.");
	}
	if (model && model.provider && model.model) {
		lines.push(`[Routing|jev] Suggested model: ${model.provider}/${model.model}.`);
	}
	lines.push(`[Routing|jev] task_type=${verdict.task} p=${verdict.p.toFixed(3)} urgency=${verdict.urgency.toFixed(2)} needs_code_model=${verdict.needsCode === undefined ? "-" : verdict.needsCode.toFixed(3)} rule=${verdict.src} (Laya System-1 × vector/regex evidence, không phải mệnh lệnh — vẫn tự phán đoán)`);
	return lines.join(" ");
}

// ---------- plugin ----------

function apply(ctx, config = {}) {
	const url = config.url || URL_DEFAULT;
	const timeoutS = Math.max(1, Math.ceil((config.timeoutMs || 2500) / 1000));
	const routingOn = config.routing !== false;
	const gateOn = config.gate === true;
	const gateThreshold = typeof config.gateThreshold === "number" ? config.gateThreshold : 0.85;
	const gateTools = new Set(Array.isArray(config.gateTools) && config.gateTools.length ? config.gateTools : ["bash", "pwsh"]);
	const maxChars = config.maxPromptChars || 2000;
	const modelByTier = config.modelByTier || {};
	const wakeOn = config.wake !== false;
	const wakeUrl = config.wakeUrl || WAKE_URL_DEFAULT;
	const coldRetryMs = typeof config.coldRetryMs === "number" ? config.coldRetryMs : 2000;
	const modelSwapOn = config.modelSwap === true;
	const verdicts = new WeakMap(); // agent -> { turn, tier } for agent/request model swap
	const centroids = buildCentroids(config.routeUtterances);

	function sleep(ms) {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	// Peak/off-peak per Coding Plan Pro v1: peak T2–T6 13:00–17:00 Asia/Ho_Chi_Minh.
	function vnPeak() {
		try {
			const parts = new Intl.DateTimeFormat("en-US", {
				timeZone: "Asia/Ho_Chi_Minh", weekday: "short", hour: "numeric", hour12: false,
			}).formatToParts(new Date());
			const get = (t) => (parts.find((p) => p.type === t) || {}).value;
			const wd = get("weekday");
			const hour = Number(get("hour"));
			const weekday = wd !== "Sat" && wd !== "Sun";
			return weekday && hour >= 13 && hour < 17;
		} catch {
			return false;
		}
	}

	const RE_QUOTA_LOW = /(hết quota|cạn quota|quota (thấp|low|hết|cạn|sắp hết)|out of quota|quota warning)/i;
	function envTags(promptText) {
		const tags = [];
		tags.push(vnPeak()
			? "[Routing|jev] PEAK Mon–Fri 13:00–17:00 VN — prefer subagent delegation; save the expensive model for decisions"
			: "[Routing|jev] OFF-PEAK — quota coefficient 0.5×, mid-size tasks are cheaper now");
		if (RE_QUOTA_LOW.test(promptText)) {
			tags.push("[Routing|jev] ⚠ LOW QUOTA — stop large tasks now, essentials only; suggest waiting for the 5h reset or off-peak hours");
		}
		return tags;
	}

	// turn-scoped routing guard: one Laya call per (agent, turn)
	const routed = new WeakMap(); // agent -> turn number already routed

	ctx.effect(function* () {
		if (wakeOn) {
			// Predictive wake: a session/agent appears -> the hybrid server reloads
			// weights in the background while the user types their first prompt.
			yield ctx.on("api-session/added", () => wake(ctx.subprocess, wakeUrl));
		}

		if (routingOn) {
			yield ctx.on("agent/pre-step", async ({ agent, messages, turn }, next) => {
				try {
					if (routed.get(agent) === turn) return next();
					const submitted = (messages || []).find((m) => m && m.source && m.source.kind === "user");
					if (!submitted) return next();
					routed.set(agent, turn);
					const flat = (submitted.content || [])
						.map((part) => (part && part.type === "text" ? part.text : ""))
						.join(" ")
						.slice(0, maxChars);
					if (!flat.trim()) return next();
					async function decideOnce(payload) {
						const r = await postDecide(ctx.subprocess, url, payload, timeoutS);
						if (r && r.wakeAccepted && coldRetryMs > 0) {
							// Hybrid cold start: server is reloading weights — wait once, retry.
							await sleep(Math.min(coldRetryMs, timeoutS * 1000));
							return postDecide(ctx.subprocess, url, payload, timeoutS);
						}
						return r;
					}
					const resp = await decideOnce(
						{ state: "User prompt: " + flat, questions: routerQuestions() });
					if (!resp || !resp.ok || !resp.answers) {
						if (resp && resp.wakeAccepted) {
							ctx.logger.warn("jev-router: laya cold (hybrid idle-unload) — wake accepted, routing skips this turn");
						}
						return next();
					}
					const pick = bestChoice(resp.answers.task_type);
					const modelTier = TIER_OF[pick.label] || "unknown";
					const verdict = {
						task: pick.label,
						p: pick.p,
						needsCode: clamp01((resp.answers.needs_code_model || {}).noul),
						urgency: typeof (resp.answers.urgency || {}).score === "number" ? resp.answers.urgency.score : -1,
					};
					const fused = fuse(modelTier, pick.p, evidenceTier(flat, centroids));
					if (fused.tier === "unknown") return next();
					verdicts.set(agent, { turn, tier: fused.tier });
					verdict.src = fused.src;
					let hint = buildHint(fused.tier, verdict, { modelByTier });
					const tags = envTags(flat).join(" ");
					if (tags) hint = hint ? `${hint} ${tags}` : tags;
					if (!hint) return next();
					ctx.logger.info(`jev-router: tier=${fused.tier} src=${fused.src} task=${verdict.task} p=${verdict.p.toFixed(3)} (turn ${turn})`);
					const decision = await next();
					if (decision && decision.kind === "enter" && Array.isArray(decision.messages)) {
						return { ...decision, messages: [...decision.messages, notice(hint, `jev route ${verdict.task}`)] };
					}
					return decision;
				} catch (error) {
					ctx.logger.warn(`jev-router: routing skipped (${error && error.message ? error.message : error})`);
					return next();
				}
			});
		}

		if (modelSwapOn) {
			// RouteLLM/Hybrid-LLM style: tier-2 turns run on the strong model for the
			// whole turn; other tiers keep whatever the user/session picked.
			yield ctx.on("agent/request", async ({ agent, turn }, next) => {
				const callConfig = await next();
				try {
					const v = verdicts.get(agent);
					if (!v || v.turn !== turn) return callConfig;
					const m = modelByTier[v.tier];
					if (!m || !m.provider || !m.model) return callConfig;
					if (callConfig && callConfig.provider === m.provider && callConfig.model === m.model) return callConfig;
					ctx.logger.info(`jev-router: model swap ${v.tier} -> ${m.provider}/${m.model} (turn ${turn})`);
					return { ...callConfig, provider: m.provider, model: m.model };
				} catch {
					return callConfig;
				}
			});
		}

		if (gateOn) {
			yield ctx.on("tools/pre-execute", async (exec, next) => {
				try {
					if (!exec || !gateTools.has(exec.name)) return next();
					const argsJson = JSON.stringify(exec.args ?? {}).slice(0, maxChars);
					if (!argsJson || argsJson === "{}") return next();
					const resp = await postDecide(ctx.subprocess, url,
						{ state: `Tool call ${exec.name}: ${argsJson}`, questions: gateQuestion(exec.name) }, timeoutS);
					const vote = gateVote((resp || {}).answers);
					if (vote === undefined) return next();
					// 2-of-3 probes hot, or mean over threshold — both from ONE batched call.
					if (vote.mean < gateThreshold && vote.hits < 2) return next();
					ctx.logger.warn(`jev-router: gate ask ${exec.name} mean=${vote.mean.toFixed(3)} max=${vote.max.toFixed(3)} hits>=0.8: ${vote.hits}/3`);
					return {
						kind: "ask",
						reason: `jev gate: tool "${exec.name}" looks risky (mean=${vote.mean.toFixed(2)}, max=${vote.max.toFixed(2)}, ${vote.hits}/3 probes ≥0.8) — user confirmation required.`,
						displayReason: {
							en: `jev gate: this "${exec.name}" call looks risky (mean=${vote.mean.toFixed(2)}, max=${vote.max.toFixed(2)}). Confirm to proceed.`,
							zh: `jev gate：此次 "${exec.name}" 调用疑似高风险（mean=${vote.mean.toFixed(2)}），需要确认。`,
						},
					};
				} catch (error) {
					ctx.logger.warn(`jev-router: gate skipped (${error && error.message ? error.message : error})`);
					return next();
				}
			});
		}
	}, "jev-router lifecycle");
}

export { apply, inject, name };
// exported for eval/run-eval.js — same code path as production routing
export { regexTier, buildCentroids, vectorTier, evidenceTier, fuse, TIER_OF, bestChoice };
