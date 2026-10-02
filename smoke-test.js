// Smoke test for @local/dsh-jev-router — runs the real plugin logic in plain
// Node against a minimal fake Cordis ctx. If the warm Laya server is up, this
// is also the end-to-end proof of "User -> Jev(Laya) -> Agent" (verdict text
// comes straight from the model).
//
//   node smoke-test.js
//   node smoke-test.js "sửa giúp hàm calculate trong utils.js"   # custom prompt
import { createRequire } from "node:module";
import { execFile } from "node:child_process";

const require = createRequire(import.meta.url);
const plugin = require("./packages/dsh-jev-router/lib/index.js");

const prompt = process.argv[2] ?? "sửa giúp hàm calculate trong utils.js, nó bị sai tổng";

// ---- minimal fake Cordis ctx ----
const registered = {};
const fakeCtx = {
	logger: { info: (...a) => console.log("   [info]", ...a), warn: (...a) => console.log("   [warn]", ...a) },
	effect(body, label) {
		console.log(`effect: ${label}`);
		const gen = typeof body === "function" ? body() : body;
		const step = (it, v) => {
			const r = it.next(v);
			if (r.done) return;
			step(it, r.value); // dispose functions are ignored in the fake
		};
		if (gen && typeof gen[Symbol.iterator] === "function") step(gen);
	},
	on(event, handler) {
		registered[event] = handler;
		return () => delete registered[event];
	},
	subprocess: {
		spawn(spec) {
			const [bin, ...args] = spec.argv;
			const out = [];
			const done = new Promise((resolve, reject) => {
				execFile(bin, args, { maxBuffer: 1 << 20, timeout: 30000 }, (err, stdout, stderr) => {
					if (err) return reject(err);
					out.push(stdout);
					void stderr;
					resolve();
				});
			});
			return {
				done,
				collected: { stdout: { readFrom: () => ({ text: out.join("") }) } },
			};
		},
	},
};

plugin.apply(fakeCtx, { routing: true, gate: false });

const preStep = registered["agent/pre-step"];
if (!preStep) throw new Error("agent/pre-step handler not registered");

const userMessage = {
	id: "fake-user-msg",
	role: "user",
	source: { kind: "user" },
	content: [{ type: "text", text: prompt }],
};

const t0 = Date.now();
const decision = await preStep(
	{ agent: {}, messages: [userMessage], turn: 1 },
	async () => ({ kind: "enter", messages: [] })
);
const elapsed = Date.now() - t0;

const notices = (decision.messages || []).filter((m) => m.source && m.source.kind === "jev-router");
console.log(`\npre-step round-trip: ${elapsed} ms`);
if (notices.length === 0) {
	console.log("RESULT: fail-open (no verdict — is the warm server ready? curl 127.0.0.1:8755/health)");
} else {
	console.log("RESULT: routed ✅");
	for (const n of notices) console.log("INJECTED NOTICE →", n.content[0].text);
}
