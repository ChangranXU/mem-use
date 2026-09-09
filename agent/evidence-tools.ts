/**
 * The judge agent's only tools: read-only access to ONE evidence folder.
 *
 * The agent never gets the built-in filesystem/bash tools — these two
 * (read / search) are the whole surface, and every path is resolved
 * inside the evidence root: absolute paths, ".." segments, and symlinks
 * pointing out of the root are refused. The judged evidence can therefore
 * never exceed what Python materialized into the folder.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";

const READ_CHAR_CAP = 200_000; // chars per read_evidence call
const SEARCH_MATCH_CAP = 200; // matched lines per search_evidence call
const SEARCH_LINE_CAP = 10_000; // chars of a line offered to the regex engine

/** Resolve `rel` inside `root`, refusing escapes; returns the real path. */
export function resolveInside(root: string, rel: string): string {
	if (typeof rel !== "string" || rel.trim() === "") {
		throw new Error("path must be a non-empty path relative to the evidence folder");
	}
	if (path.isAbsolute(rel) || /^[a-zA-Z]:[\\/]/.test(rel)) {
		throw new Error(`absolute paths are not allowed (${rel}); use a path relative to the evidence folder`);
	}
	const segments = rel.split(/[\\/]+/);
	if (segments.some((s) => s === "..")) {
		throw new Error(`".." segments are not allowed (${rel})`);
	}
	const rootReal = fs.realpathSync(root);
	const target = path.resolve(rootReal, ...segments);
	// realpath resolves symlinks: a link pointing out of the root fails the
	// prefix check below instead of leaking its target
	const targetReal = fs.realpathSync(target);
	if (targetReal !== rootReal && !targetReal.startsWith(rootReal + path.sep)) {
		throw new Error(`${rel} resolves outside the evidence folder`);
	}
	return targetReal;
}

/** Sorted relative paths of every regular file under `dirRoot` (symlinks skipped). */
function walkFiles(rootReal: string, rel = ""): string[] {
	const out: string[] = [];
	const dir = rel ? path.join(rootReal, rel) : rootReal;
	let entries: fs.Dirent[];
	try {
		entries = fs.readdirSync(dir, { withFileTypes: true });
	} catch {
		return out; // unreadable subfolder: listing what we can beats failing
	}
	for (const e of entries) {
		const childRel = rel ? `${rel}/${e.name}` : e.name;
		if (e.isDirectory()) {
			out.push(...walkFiles(rootReal, childRel));
		} else if (e.isFile()) {
			out.push(childRel);
		} // symlinks and specials are skipped, never followed
	}
	return out.sort();
}

const readSchema = Type.Object({
	path: Type.String({ description: "Evidence file path relative to the evidence folder, e.g. steps/step_000012.md" }),
	offset: Type.Optional(
		Type.Integer({ minimum: 1, description: "First line to return (1-based); default 1" }),
	),
	limit: Type.Optional(
		Type.Integer({ minimum: 1, description: "Maximum number of lines to return" }),
	),
});

const searchSchema = Type.Object({
	pattern: Type.String({ description: "Regular expression to match (per line)" }),
	path: Type.Optional(
		Type.String({ description: "Search this file or subfolder (relative path); default steps" }),
	),
	ignore_case: Type.Optional(Type.Boolean({ description: "Case-insensitive matching (default false)" })),
});

export function createEvidenceTools(root: string): AgentTool<any>[] {
	const readEvidence: AgentTool<typeof readSchema, { path: string }> = {
		label: "Read evidence",
		name: "read_evidence",
		description:
			"Read one evidence file with 1-based line numbers. Use offset/limit to inspect the context around search hits; line numbers are not agent-step anchors.",
		parameters: readSchema,
		execute: async (_id, params) => {
			const target = resolveInside(root, params.path);
			const text = fs.readFileSync(target, "utf8");
			const lines = text.split("\n");
			const from = (params.offset ?? 1) - 1;
			if (from > 0 && from >= lines.length) {
				return {
					content: [{ type: "text", text: `(the file has ${lines.length} lines; offset ${params.offset} is past its end)` }],
					details: { path: params.path },
				};
			}
			const selected = params.limit ? lines.slice(from, from + params.limit) : lines.slice(from);
			let body = selected.map((line, i) => `${from + i + 1}: ${line}`).join("\n");
			let note = "";
			if (body.length > READ_CHAR_CAP) {
				body = body.slice(0, READ_CHAR_CAP);
				note = `\n(... cut at ${READ_CHAR_CAP} chars; call again with an offset/limit to read the rest)`;
			}
			return {
				content: [{ type: "text", text: body + note }],
				details: { path: params.path },
			};
		},
	};

	const searchEvidence: AgentTool<typeof searchSchema, { pattern: string; path?: string }> = {
		label: "Search evidence",
		name: "search_evidence",
		description:
			"Search a regular expression in the recorded steps (or one evidence file/subfolder via `path`). Returns `path:line: match excerpt`; read the surrounding lines before citing. No matches alone do not prove no influence.",
		parameters: searchSchema,
		execute: async (_id, params) => {
			let regex: RegExp;
			try {
				regex = new RegExp(params.pattern, params.ignore_case ? "i" : "");
			} catch (e) {
				throw new Error(`invalid regular expression ${params.pattern}: ${e instanceof Error ? e.message : e}`);
			}
			const rootReal = fs.realpathSync(root);
			let files: string[];
			if (params.path) {
				const scope = resolveInside(root, params.path);
				const rel = path.relative(rootReal, scope).split(path.sep).join("/");
				files = fs.statSync(scope).isDirectory()
					? walkFiles(rootReal).filter((f) => !rel || f.startsWith(rel + "/"))
					: [rel];
			} else {
				files = walkFiles(rootReal).filter((f) => f.startsWith("steps/"));
			}
			const hits: string[] = [];
			let truncated = false;
			for (const f of files) {
				if (hits.length >= SEARCH_MATCH_CAP) {
					truncated = true;
					break;
				}
				const text = fs.readFileSync(path.join(rootReal, f), "utf8");
				const lines = text.split("\n");
				for (let i = 0; i < lines.length; i++) {
					const match = regex.exec(lines[i].slice(0, SEARCH_LINE_CAP));
					if (match) {
						// Long reasoning paragraphs put the actual match well past
						// the head of the line; show it, not an unrelated prefix.
						const start = Math.max(0, match.index - 100);
						const end = Math.max(start + 300, match.index + match[0].length);
						const excerpt = (start ? "…" : "") + lines[i].slice(start, end)
							+ (end < lines[i].length ? "…" : "");
						hits.push(`${f}:${i + 1}: ${excerpt}`);
						if (hits.length >= SEARCH_MATCH_CAP) {
							truncated = true;
							break;
						}
					}
				}
			}
			let textOut = hits.join("\n") || "(no matches)";
			if (truncated) {
				textOut += `\n(... cut at ${SEARCH_MATCH_CAP} matched lines; narrow the pattern or scope to see more)`;
			}
			return { content: [{ type: "text", text: textOut }], details: { pattern: params.pattern, path: params.path } };
		},
	};

	return [readEvidence, searchEvidence];
}
