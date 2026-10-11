/**
 * Trajectory recording for one judge-agent conversation, pi-style: the
 * agent's event stream is subscribed and every completed message is
 * appended to a JSONL file as it happens — including turns the
 * context-budget later rolls back (recorded as tombstones naming the
 * removed lines), the injected forced-answer message (which fires no
 * event, since the budget injects it by context rewrite), and the
 * per-attempt diagnostics at the end.
 * Records are masked before serialization. The live messages stay unchanged.
 *
 * `seq` is attempt-local: each runner process starts its counter at 0
 * (an `attempt` marker line opens the process's lines), and tombstone
 * references never cross an attempt boundary because a rollback only
 * removes the current attempt's messages.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";

export interface TrajectoryRecord {
	seq: number;
	type: string;
	[key: string]: unknown;
}

export class TrajectoryLog {
	private seq = 0;
	/** the messages appended so far, in log order — the seq source for
	 * tombstones (the only removals are this process's own rollbacks). */
	private logged: { message: AgentMessage; seq: number }[] = [];

	private readonly filePath: string;
	private readonly mask: (value: unknown) => unknown;

	constructor(filePath: string, mask: (value: unknown) => unknown) {
		// a plain field assignment, not a parameter property: node's
		// type-stripping mode rejects non-erasable syntax at runtime
		this.filePath = filePath;
		this.mask = mask;
		fs.mkdirSync(path.dirname(filePath), { recursive: true });
	}

	private append(record: Omit<TrajectoryRecord, "seq">): void {
		fs.appendFileSync(this.filePath, JSON.stringify(this.mask({ seq: this.seq++, ...record })) + "\n");
	}

	attempt(n: number): void {
		this.append({ type: "attempt", attempt: n });
	}

	message(message: AgentMessage): void {
		const at = this.seq;
		this.append({ type: "message", message });
		this.logged.push({ message, seq: at });
	}

	/** Tombstone the last `count` logged messages as removed from the live
	 * conversation (a context-budget rollback or a forced-round tool tail). */
	tombstoneTail(count: number, reason: string): void {
		const removed = this.logged.splice(Math.max(0, this.logged.length - count), count);
		if (removed.length > 0) {
			this.append({ type: "tombstone", removedSeqs: removed.map((e) => e.seq), reason });
		}
	}

	attemptEnd(diagnostics: Record<string, unknown>): void {
		this.append({ type: "attempt_end", ...diagnostics });
	}
}
