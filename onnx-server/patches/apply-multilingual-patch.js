// Idempotent postinstall patch: teach @receptron/laya to read special tokens
// from the bundle's tokenizer/tokenizer_config.json instead of hardcoding
// BERT's [CLS]/[SEP]/[MASK]/[PAD]. Required for multilingual (mmBERT)
// checkpoints — see README §13.
//
// Run: node patches/apply-multilingual-patch.js   (wired as postinstall)
import { readFileSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const target = join(here, "..", "node_modules", "@receptron/laya", "dist", "laya.js");

const ORIGINAL =
	'        const ids = { cls: id("[CLS]"), sep: id("[SEP]"), mask: id("[MASK]"), pad: id("[PAD]"), maskTok: "[MASK]" };';
const PATCHED = [
	'        const tokCfg = (await read("tokenizer/tokenizer_config.json"));',
	"        // PATCH(jev-laya): mmBERT/multilingual checkpoints use sentencepiece-style",
	"        // specials (<bos>/<eos>/<pad>/<mask>) — read them from tokenizer_config.json",
	"        // instead of hardcoding BERT's [CLS]/[SEP]/[MASK]/[PAD].",
	'        const tokName = (k, fb) => (typeof tokCfg?.[k] === "string" ? tokCfg[k] : fb);',
	'        const ids = { cls: id(tokName("cls_token", "[CLS]")), sep: id(tokName("sep_token", "[SEP]")), mask: id(tokName("mask_token", "[MASK]")), pad: id(tokName("pad_token", "[PAD]")), maskTok: tokName("mask_token", "[MASK]") };',
].join("\n");

let src;
try {
	src = readFileSync(target, "utf8");
} catch {
	console.log("[jev-patch] @receptron/laya chưa cài — bỏ qua (postinstall sẽ chạy lại sau npm install)");
	process.exit(0);
}

if (src.includes("PATCH(jev-laya")) {
	console.log("[jev-patch] đã vá từ trước — skip");
	process.exit(0);
}
if (!src.includes(ORIGINAL)) {
	console.error("[jev-patch] CẢNH BÁO: không tìm thấy dòng gốc để vá (upstream có thể đã đổi) — kiểm tra tay:");
	console.error(`  ${target}`);
	process.exit(0); // không phá npm install
}
writeFileSync(target, src.replace(ORIGINAL, PATCHED));
console.log("[jev-patch] đã vá specials cho checkpoint multilingual ✓");
