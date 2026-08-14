"""Shared synthetic-fixture builder for schema-v6 annotated trajectories.

Fixtures mirror the annotation protocol emitted by traj-recorder schema v6
(see extension/traj-recorder README "Annotating memory actions"): a
memory_session, role bindings, generation operations (start / change / end),
search operations, and bound deliveries — plus the ordinary turn/message/
tool events the step stream consumes.
"""

import hashlib
import json

MAIN = "MAIN"
MEM = "EXTRACT"
SID = "11111111-2222-4333-8444-555555555555"
TASK = "<pr_description>fix the bug</pr_description>"


def ev(seq, type, **kw):
    return {"seq": seq, "type": type, **kw}


def cure_ref(key, value="v", memory_type="insight", confidence=0.9,
             review_status="approved", store_id=1, version_id=None, namespace="ns"):
    """One CURE-row MemoryRef as emitted by the bridge's annotations.

    Like the real bridge, the version_id digests the row's semantic content
    (a value change is a new version) and the inline content carries its
    sha256; confidence/review metadata stay outside the identity. The
    namespace names the store (the bridge hashes its DB path into it).
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {
        "version_id": version_id or f"{store_id}:{digest[:16]}",
        "identity_strength": "native_stable",
        "identity_scheme": "cure-sqlite-row-version-v1",
        "item_id": f"item-{key}",
        "namespace": namespace,
        "content": {"availability": "inline", "text": value, "sha256": digest},
        "extensions": {
            "cure": {
                "store_id": store_id,
                "memory_type": memory_type,
                "key": key,
                "confidence": confidence,
                "review_status": review_status,
                "needs_verification": False,
            }
        },
    }


class AnnotatedBuilder:
    """Assembles a synthetic annotated (schema v6) trajectory.jsonl."""

    def __init__(self, main=MAIN, mem=MEM, adapter=None, task_ref=None):
        self.events = []
        self.seq = 0
        self.turn = 0
        self.main = main
        self.mem = mem
        self.adapter = adapter or {"name": "cure", "version": "0.1.0"}
        self._n_op = 0
        self.last_call = None
        self._emit("agent_start")
        self.annotation(
            "memory_session",
            {
                "trace_session_id": SID,
                "session_kind": "episode",
                "adapter": self.adapter,
                "instance_id": "pydata__xarray-2905",
                "task": task_ref or {"availability": "inline", "text": TASK},
            },
            lane=self.main,
        )
        self.annotation(
            "memory_role_bind",
            {"trace_session_id": SID, "logical_role": "main", "adapter": self.adapter},
            lane=self.main,
        )
        self.annotation(
            "memory_role_bind",
            {"trace_session_id": SID, "logical_role": "memory", "adapter": self.adapter},
            lane=self.mem,
        )

    def _emit(self, type, **kw):
        self.seq += 1
        self.events.append(ev(self.seq, type, **kw))
        return self

    def annotation(self, type, payload, lane=None, binding_status="not_requested",
                   call_refs=None, terminal_call_ref=None):
        self.seq += 1
        event = ev(
            self.seq, type,
            annotation_id=f"00000000-0000-4000-8000-{self.seq:012d}",
            binding_status=binding_status,
            call_refs=call_refs or [],
            proxy_source=lane,
            payload=payload,
        )
        if terminal_call_ref is not None:
            event["terminal_call_ref"] = terminal_call_ref
        self.events.append(event)
        return self

    def agent_turn(self, with_task=False, reasoning="thinking", command="ls", result="ok"):
        """One main-lane turn with one bash tool call; its call ref lands in
        `self.last_call` (deliveries bind to it)."""
        if with_task:
            self._emit("message_end", proxy_source=self.main, role="user", msg=1, content=TASK)
        self.turn += 1
        call_id = f"{self.turn:06d}-x"
        self._emit("turn_start", turn=self.turn, call_id=call_id, proxy_source=self.main)
        self._emit("message_end", proxy_source=self.main, role="assistant", msg=2,
                   turn=self.turn, content="", reasoning_content=reasoning,
                   tool_calls=[{"id": call_id, "name": "bash", "args": {"command": command}}],
                   finish_reason="tool_calls")
        self._emit("tool_execution_start", proxy_source=self.main, turn=self.turn,
                   tool_call_id=call_id, name="bash", args={"command": command})
        if result is not None:
            self._emit("tool_execution_end", proxy_source=self.main, turn=self.turn, msg=3,
                       tool_call_id=call_id, name="bash", status="completed", result=result)
        self._emit("turn_end", turn=self.turn, status="completed", proxy_source=self.main)
        self.last_call = {"call_sequence": self.turn, "call_id": call_id,
                          "role_call_index": self.turn}
        return self

    def memory_turn(self):
        """One memory-lane LLM turn (the extraction decision call)."""
        self.turn += 1
        call_id = f"{self.turn:06d}-m"
        self._emit("turn_start", turn=self.turn, call_id=call_id, proxy_source=self.mem)
        self._emit("message_end", proxy_source=self.mem, role="assistant",
                   msg=2, turn=self.turn, content="{}")
        self._emit("turn_end", turn=self.turn, status="completed", proxy_source=self.mem)
        self.last_call = {"call_sequence": self.turn, "call_id": call_id,
                          "role_call_index": 1}
        return self

    def rewrite(self, src=MAIN, kept=5, dropped=1, replay_tool_ends=True):
        """A history_rewritten plus what may follow it turn-less on the agent
        lane: the gate's origin-history re-record of the response-side
        assistant message, and (at-least-once retry) a duplicate close of an
        earlier tool execution — deduped downstream like any re-closing."""
        self._emit("history_rewritten", proxy_source=src, kept=kept, dropped=dropped)
        self._emit("message_end", proxy_source=self.main, role="assistant", msg=2,
                   content="", reasoning_content="replayed", origin="history")
        if replay_tool_ends:
            self._emit("tool_execution_end", proxy_source=self.main, msg=3,
                       tool_call_id="000001-x", name="bash", status="completed",
                       result="replayed-result")
        return self

    def _op_header(self):
        self._n_op += 1
        return {
            "trace_session_id": SID,
            "operation_id": f"op-{self._n_op:04d}",
            "requested_by": "main",
            "handled_by": "memory",
        }

    def generate(self, produced=(), rejections=None, status="completed",
                 extraction_step="10", changes=(), error_codes=(), bind=None):
        """One generation operation (start → changes → end); returns its
        operation_id (e.g. for a delivery naming it by mistake)."""
        header = self._op_header()
        self.annotation("memory_generate_start",
                        {**header, "trigger": "step", "inputs": []}, lane=self.mem)
        for i, (action, before, after) in enumerate(changes):
            self.annotation(
                "memory_change",
                {
                    "trace_session_id": SID,
                    "operation_id": header["operation_id"],
                    "change_id": f"chg-{i}",
                    "action": action,
                    "before": list(before),
                    "after": list(after),
                    "relationships": [],
                    "evidence": "observed_diff",
                    "completeness": "complete",
                    "change_index": i,
                    "change_count": len(changes),
                },
                lane=self.mem,
                binding_status="bound" if bind else "not_requested",
                call_refs=[bind] if bind else [],
                terminal_call_ref=bind,
            )
        self.annotation(
            "memory_generate_end",
            {
                **header,
                "status": status,
                "produced": list(produced),
                "change_count": len(changes),
                "state_evidence": "complete",
                "error_codes": list(error_codes),
                "extensions": {
                    "cure": {
                        "extraction_step": extraction_step,
                        "candidates": len(produced),
                        "rejected": sum((rejections or {}).values()),
                        "rejected_by_reason": dict(rejections or {}),
                    }
                },
            },
            lane=self.mem,
            binding_status="bound" if bind else "not_requested",
            call_refs=[bind] if bind else [],
            terminal_call_ref=bind,
        )
        return header["operation_id"]

    def search(self, returned=(), query="the task query", with_end=True,
               matched_count=None):
        """One search operation; returns its operation_id for deliveries.
        `matched_count` is the protocol's {value, precision} field (None =
        not recorded)."""
        header = self._op_header()
        self.annotation(
            "memory_search_start",
            {**header, "query": {"availability": "inline", "text": query}},
            lane=self.mem,
        )
        if with_end:
            payload = {
                **header,
                "status": "completed",
                "returned": [
                    {"ordinal": i, "memory": ref} for i, ref in enumerate(returned)
                ],
                "error_codes": [],
            }
            if matched_count is not None:
                payload["matched_count"] = matched_count
            self.annotation("memory_search_end", payload, lane=self.mem)
        return header["operation_id"]

    def delivery(self, search_operation_id, memories=(), bind=None, main_step=None,
                 status="placed", content="## CURE Memory Policy\n…"):
        """One memory_delivery on the main lane (bound when `bind` is given);
        content=None records a redacted placement (no text, no chars)."""
        ref = ({"availability": "redacted", "reason": "policy"} if content is None
               else {"availability": "inline", "text": content, "chars": len(content)})
        payload = {
            "trace_session_id": SID,
            "delivery_id": f"d-{self.seq:04d}",
            "search_operation_id": search_operation_id,
            "from_role": "memory",
            "to_role": "main",
            "status": status,
            "placement": {
                "kind": "prompt_message",
                "content": ref,
                "proof_kind": "canonical_message",
            },
            "memories": list(memories),
        }
        if main_step is not None:
            payload["main_step"] = main_step
        self.annotation(
            "memory_delivery", payload, lane=self.main,
            binding_status="bound" if bind else "not_requested",
            call_refs=[bind] if bind else [],
            terminal_call_ref=bind,
        )
        return self

    def recall(self, ref=None, query="the task query", content="## CURE Memory Policy\n…"):
        """Convenience: search + agent turn + bound delivery (one injection).
        `ref` may be a single MemoryRef or a list of them."""
        refs = list(ref) if isinstance(ref, (list, tuple)) else [
            ref or cure_ref("key_a", "Some value here")
        ]
        sid = self.search(returned=refs, query=query)
        self.agent_turn()
        self.delivery(sid, memories=refs, bind=self.last_call, content=content)
        return self

    def write(self, directory, run_id="test-run", task_preview="fix the bug"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self.events) + "\n", encoding="utf-8"
        )
        (directory / "run.json").write_text(json.dumps(
            {"run_id": run_id, "model_name": "test-model",
             "started_at": "2026-01-01T00:00:00Z", "status": "stopped",
             "task": task_preview}
        ), encoding="utf-8")
        return directory


def parse(b: AnnotatedBuilder, tmp_path, name="t1"):
    d = b.write(tmp_path / name)
    warnings = []
    from eval.memgraph.parse_trajectory import parse_trajectory
    traj, ops, _events = parse_trajectory(d, warnings)
    return traj, ops, warnings
