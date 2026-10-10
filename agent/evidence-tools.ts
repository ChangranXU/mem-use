/**
 * The judge agent's only tools: read-only access to ONE evidence folder.
 *
 * The agent never gets the built-in filesystem/bash tools — these two
 * (read / search) are the whole surface, and every path is resolved
 * inside the evidence root: absolute paths, ".." segments, and symlinks
 * pointing out of the root are refused. The judged evidence can therefore
 * never exceed what Python materialized into the folder.
 *
 * A judgment needs whole files (Python certifies complete reads), and every
 * model call resends the conversation, so read_evidence takes several paths
 * at once: a footprint window costs one call instead of one per step.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as vm from "node:vm";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";

const READ_CHAR_CAP = 200_000; // chars per read_evidence call, across its files
const SEARCH_MATCH_CAP = 200; // matched lines per search_evidence call
const SEARCH_TIMEOUT_MS = 5_000; // a synchronous regex can backtrack without bound on a long line

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

/** What one returned block covers; Python certifies complete reads from it. */
export interface FileRead {
	path: string;
	lines_read: number;
	line_start?: number;
	line_end?: number;
	truncated?: boolean;
	error?: string;
}

const readSchema = Type.Object({
	paths: Type.Optional(Type.Array(Type.String(), {
		minItems: 1,
		description: "Evidence files to read whole, relative to the evidence folder, e.g. [\"task.md\", \"steps/step_000012.md\", \"steps/step_000013.md\"]",
	})),
	path: Type.Optional(Type.String({ description: "One evidence file (same as a one-entry paths)" })),
	offset: Type.Optional(
		Type.Integer({ minimum: 1, description: "First line to return (1-based); default 1" }),
	),
	limit: Type.Optional(
		Type.Integer({ minimum: 1, description: "Maximum number of lines to return" }),
	),
	column: Type.Optional(
		Type.Integer({ minimum: 1, description: "First Unicode character in the offset line (1-based); use the continuation position from a truncated read" }),
	),
});

const searchSchema = Type.Object({
	pattern: Type.String({ description: "JavaScript regular expression matched per line; use ignore_case, not inline (?i)" }),
	path: Type.Optional(
		Type.String({ description: "Search this file or subfolder (relative path); default steps" }),
	),
	ignore_case: Type.Optional(Type.Boolean({ description: "Case-insensitive matching (default false)" })),
});

/** One file's numbered lines within `room` characters, plus where to continue. */
function readLines(rel: string, text: string, room: number, offset = 1, limit?: number, column = 1):
	{ block: string; used: number; read: FileRead } {
	const lines = text.split("\n");
	const from = offset - 1;
	if (from > 0 && from >= lines.length) {
		const block = `(the file has ${lines.length} lines; offset ${offset} is past its end)`;
		return { block, used: block.length, read: { path: rel, lines_read: 0 } };
	}
	const selected = limit ? lines.slice(from, from + limit) : lines.slice(from);
	const returned: string[] = [];
	let used = 0, note = "";
	for (let i = 0; i < selected.length; i++) {
		const line = from + i + 1, col = i === 0 ? column : 1;
		const body = col === 1 ? selected[i] : Array.from(selected[i]).slice(col - 1).join("");
		const prefix = col === 1 ? `${line}: ` : `${line}@${col}: `;
		const space = room - used - prefix.length - (returned.length ? 1 : 0);
		if (space <= 0) {
			note = `\n(... character cap; continue with path=${rel}, offset=${line}, column=${col})`;
			break;
		}
		let chunk = body.slice(0, space);
		// Never split a surrogate pair: continuation columns count Unicode
		// characters, matching Python's evidence-coverage coordinates.
		if (chunk.length < body.length && /[\uD800-\uDBFF]$/.test(chunk)) chunk = chunk.slice(0, -1);
		returned.push(prefix + chunk);
		used += prefix.length + chunk.length + (returned.length > 1 ? 1 : 0);
		if (chunk.length < body.length) {
			note = `\n(... character cap; continue with path=${rel}, offset=${line}, column=${col + Array.from(chunk).length})`;
			break;
		}
	}
	const truncated = !!note;
	// A limit that stops early must say so: a clean slice boundary reads
	// like the end of the file, and coverage needs every non-blank line.
	const next = from + selected.length, last = lines.findLastIndex((line) => line.trim());
	if (!note && last >= next) {
		const more = last + 1 - next;
		note = `\n(... ${more} more line${more === 1 ? "" : "s"}; continue with path=${rel}, offset=${next + 1})`;
	}
	return {
		block: returned.join("\n") + note,
		used,
		// Coverage describes only text returned, never the selected tail
		// removed by the character cap. The final line may be partial.
		read: { path: rel, lines_read: returned.filter((line) => line.replace(/^\d+(?:@\d+)?: /, "").trim()).length,
			line_start: from + 1, line_end: from + returned.length, truncated },
	};
}

export function createEvidenceTools(root: string): AgentTool<any>[] {
	const readEvidence: AgentTool<typeof readSchema, { files: FileRead[] }> = {
		label: "Read evidence",
		name: "read_evidence",
		description:
			"Read evidence files with 1-based line numbers. Pass every file you need in `paths` "
			+ "(for example all steps of a footprint window) to read them in one call; offset/limit/column "
			+ "page through each named file. Line numbers are not agent-step anchors.",
		parameters: readSchema,
		execute: async (_id, params) => {
			if (params.paths && params.path) throw new Error("pass either path or paths, not both");
			const targets = params.paths ?? (params.path !== undefined ? [params.path] : []);
			if (!targets.length) throw new Error("name the files to read in paths");
			const rootReal = fs.realpathSync(root);
			const blocks: string[] = [], files: FileRead[] = [];
			let used = 0;
			for (const rel of targets) {
				let text: string, relative: string;
				try {
					const target = resolveInside(root, rel);
					relative = path.relative(rootReal, target).split(path.sep).join("/");
					if (used >= READ_CHAR_CAP) {
						blocks.push(`==> ${relative} <==\n(not read: the call's character cap was reached; read it in another call)`);
						files.push({ path: relative, lines_read: 0, truncated: true });
						continue;
					}
					text = fs.readFileSync(target, "utf8");
				} catch (e) {
					const error = e instanceof Error ? e.message : String(e);
					blocks.push(`==> ${rel} <==\n(error: ${error})`);
					files.push({ path: rel, lines_read: 0, error });
					continue;
				}
				const result = readLines(relative, text, READ_CHAR_CAP - used, params.offset, params.limit, params.column);
				used += result.used;
				blocks.push(`==> ${relative} <==\n${result.block}`);
				files.push(result.read);
			}
			// Every file failing is a failed call; a batch keeps its good reads.
			if (files.every((file) => file.error)) throw new Error(files.map((file) => file.error).join("; "));
			// One text block per file, aligned with details.files.
			return { content: blocks.map((text) => ({ type: "text" as const, text })), details: { files } };
		},
	};

	const searchEvidence: AgentTool<typeof searchSchema, { pattern: string; path?: string }> = {
		label: "Search evidence",
		name: "search_evidence",
		description:
			"Search a regular expression in the recorded steps (or one evidence file/subfolder via `path`). "
			+ "Returns `path:line: match excerpt`; use it to locate identifiers, then read the whole files "
			+ "before citing. No matches alone do not prove no influence.",
		parameters: searchSchema,
		execute: async (_id, params) => {
			let regex: RegExp;
			try {
				regex = new RegExp(params.pattern, params.ignore_case ? "i" : "");
			} catch (e) {
				throw new Error(`invalid regular expression ${params.pattern}: ${e instanceof Error ? e.message : e}`);
			}
			const rootReal = fs.realpathSync(root);
			// The default scope passes the same boundary check as an explicit one:
			// a symlinked steps/ must not lead the walk outside the folder.
			const scope = resolveInside(root, params.path || "steps");
			const rel = path.relative(rootReal, scope).split(path.sep).join("/");
			const files = fs.statSync(scope).isDirectory() ? walkFiles(rootReal, rel) : [rel];
			const hits: string[] = [];
			let truncated = false;
			const search = () => {
				for (const f of files) {
					const text = fs.readFileSync(path.join(rootReal, f), "utf8");
					const lines = text.split("\n");
					for (let i = 0; i < lines.length; i++) {
						const match = regex.exec(lines[i]);
						if (match) {
							// Look ahead one match so exactly 200 matches do not claim a hidden tail.
							if (hits.length === SEARCH_MATCH_CAP) {
								truncated = true;
								break;
							}
							// Long reasoning paragraphs put the actual match well past
							// the head of the line; show it, not an unrelated prefix.
							const start = Math.max(0, match.index - 100);
							const end = start + 300;
							const excerpt = (start ? "…" : "") + lines[i].slice(start, end)
								+ (end < lines[i].length ? "…" : "");
							hits.push(`${f}:${i + 1}: ${excerpt}`);
						}
					}
					if (truncated) break;
				}
			};
			// vm's timeout interrupts the regex engine itself, so a pathological
			// pattern costs the judge one tool error instead of the whole run.
			try {
				vm.runInNewContext("search()", { search }, { timeout: SEARCH_TIMEOUT_MS });
			} catch (e) {
				if ((e as { code?: string }).code !== "ERR_SCRIPT_EXECUTION_TIMEOUT") throw e;
				throw new Error(`search exceeded ${SEARCH_TIMEOUT_MS / 1000}s; use a pattern without unbounded `
					+ "repetition such as .* or a narrower path");
			}
			let textOut = hits.join("\n") || "(no matches)";
			if (truncated) {
				textOut += `\n(... more than ${SEARCH_MATCH_CAP} matched lines; narrow the pattern or path)`;
			}
			return { content: [{ type: "text", text: textOut }], details: { pattern: params.pattern, path: params.path } };
		},
	};

	return [readEvidence, searchEvidence];
}
