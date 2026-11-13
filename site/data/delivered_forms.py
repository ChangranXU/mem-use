"""Add per-form breakdowns to results.json from the retained injudge reports.

Forms follow the paper's delivered-form table (tab:delivered-forms). Reports
carry no form field, so a form comes from the span's placement slot in its
prepared evidence and the entry's key shape. Profile axes are means of
per-span entry shares, so they split exactly by form. A span verdict is
credited to the entries that decided it; delivered tokens split by exposed
characters. USD cost cannot be split by form and is not. Verdict and profile
splits are also kept per run and window of consecutive tasks.

Usage: uv run python site/data/delivered_forms.py [batch.json]
The batch and its reports are read only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from injudge.profiles import span_profile  # noqa: E402
from run_statistics import MEMORAX, MEMORAX_REPORTS, TASK_WINDOW, task_positions  # noqa: E402
from util import atomic_write  # noqa: E402

BATCH = Path("/Volumes/T9/eval-result/batches/20261023-091032-b21f276d-gold-resume/batch.json")
RESULTS = Path(__file__).with_name("results.json")
ROWS = {
    ("memgovern", None): "swe-memgovern",
    ("mem0", "local_global-enabled"): "swe-mem0-local-global",
    ("mem0", "local_global-disabled"): "swe-mem0-local",
    ("tencentdb-proxy", "native_short-timers"): "swe-tencentdb",
}
FORMS = {
    "memgovern": [("search_preview", "Search preview", "A"), ("full_card", "Full card", "A")],
    "mem0": [("task_recall", "Task recall", "H"), ("search_result", "Search result", "A")],
    "tencentdb-proxy": [
        ("skill_catalog", "Skill catalog", "H"), ("persona", "Persona", "H"),
        ("scene_index", "Scene index", "H, A"), ("skill_search_preview", "Skill search preview", "A"),
        ("full_skill", "Full skill", "A"), ("scene_content", "Scene content", "A"),
    ],
}
# Copies of still-visible entries are judged automatically, never by the agent judge.
COPY = ("copy", "Copy of visible entries", "H, A")
OTHER = ("other", "Empty or unclassified", "")
AXES = {"relevance": "relevance", "corroboration": "corroboration", "information_gain": "info_gain",
        "acted_on": "influence", "benefit": "benefit", "harm": "harm"}
SLOT = re.compile(r"Presentation slot '([^']+)'")


def placements(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Map span id to (slot, first placement line) from the prepared evidence."""
    out = {}
    for span in (run_dir / "evidence").glob("*/span.md"):
        text = span.read_text(errors="replace")
        slot = SLOT.search(text)
        first = text.split("DELIVERED PLACEMENT (verbatim):\n", 1)[1].split("\n", 1)[0] if "DELIVERED PLACEMENT" in text else ""
        out[span.parent.name.rsplit("-", 1)[0].replace("_", ":")] = (slot.group(1) if slot else "", first)
    return out


def form(system: str, slot: str, first: str, key: str) -> str:
    if system == "memorax":  # every judged span is an agent-directed search result
        return "search_result"
    if system == "memgovern":
        return "full_card" if first.startswith("=") else "search_preview" if slot else "other"
    if system == "mem0":
        return {"system_suffix": "task_recall", "search_result": "search_result"}.get(slot, "other")
    if key == "persona.md#injected":
        return "persona"
    if key.startswith("available_skills/"):
        return "skill_catalog"
    if key.startswith("skl-"):
        return "skill_search_preview" if key.endswith(("#index", "#snippet")) else "full_skill"
    if key.endswith("#index"):
        return "scene_index"
    return "scene_content" if slot else "other"


def deciding(op: dict, sub: dict[str, dict]) -> set[str]:
    """Entries that set the span verdict; a neutral span splits over all entries."""
    verdict = op["verdict"]
    keys = {"positive": ("benefit",), "negative": ("harm",), "unresolved": ("unknown_influence", "unknown_effect")}.get(verdict)
    chosen = {i for i, counts in sub.items() if keys and any(counts[k] for k in keys)}
    return chosen or set(sub)


def entry_flags(op: dict) -> dict[str, dict]:
    """Per-entry flag counts, reusing span_profile on one-entry views of the span."""
    out = {}
    for memory in op.get("memories") or []:
        one = {**op, "memories": [memory], "rubric": {memory["id"]: (op.get("rubric") or {}).get(memory["id"])}}
        profile = span_profile(one) if one["rubric"][memory["id"]] else None
        out[memory["id"]] = profile["counts"] if profile else None
    return out


def within_form(flags: dict[str, dict], forms: dict[str, str]) -> dict[str, float]:
    """A mixed span gives each present form its own entry denominator."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for entry, counts in flags.items():
        if counts is not None:
            grouped[forms[entry]].append(counts["benefit"])
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def main() -> None:
    batch = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else BATCH).read_text())
    split = lambda: {"verdicts": defaultdict(lambda: defaultdict(float)),  # noqa: E731
                     "profile": defaultdict(lambda: defaultdict(float)), "profiled": 0,
                     "within_form": defaultdict(lambda: {"spans": 0, "sum": 0.0})}
    acc = defaultdict(lambda: {**split(), "tokens": defaultdict(float), "unclassified": 0, "runs": {}})
    dataset = json.loads(RESULTS.read_text())
    tasks = {row["id"]: row["tasks_per_run"] for row in dataset["rows"]}
    sources = [(run["integration"], ROWS[(run["integration"], run["setting"])], run["run"], Path(run["output"])) for run in batch["runs"]]
    sources += [("memorax", "swe-memorax", path.split("/")[0], MEMORAX / path) for path in MEMORAX_REPORTS]
    systems = {row: system for system, row, _, _ in sources}
    for system, row, run, out in sources:
        report = json.loads((out / "injudge-report.json").read_text())
        slots, a, position = placements(out), acc[row], task_positions(report, tasks[row])
        windows = a["runs"][run] = defaultdict(split)
        for op in report["operations"]:
            parts = (a, windows[position[op["episode_execution_id"]] // TASK_WINDOW])
            memories = op.get("memories") or []
            slot, first = slots.get(op["operation_id"], ("", ""))
            if op.get("auto") == "no_eligible_step":
                forms = {m["id"]: "copy" for m in memories}
            else:
                forms = {m["id"]: form(system, slot, first, m["key"]) for m in memories}
            if op.get("auto") is None and "other" in forms.values():
                a["unclassified"] += 1
            exposure = op.get("memory_exposure") or []
            chars = {m["id"]: (exposure[i].get("exposed_chars") if i < len(exposure) else None) or len(m.get("value") or "")
                     for i, m in enumerate(memories)}
            tokens = (op.get("delivery_cost") or {}).get("approx_tokens_total") or 0
            total = sum(chars.values())
            for i, n in chars.items():
                a["tokens"][forms[i]] += tokens * n / total if total else 0
            if not memories:
                a["tokens"]["other"] += tokens
            if op.get("auto") == "empty_delivery" or op.get("verdict") not in ("positive", "negative", "neutral", "unresolved"):
                continue
            flags = entry_flags(op)
            whole = span_profile(op)
            chosen = deciding(op, {i: c for i, c in flags.items() if c}) if whole else set(forms)
            for part in parts:
                if whole:
                    part["profiled"] += 1
                    for key, value in within_form(flags, forms).items():
                        part["within_form"][key]["spans"] += 1
                        part["within_form"][key]["sum"] += value
                    for i, counts in flags.items():
                        for axis, key in AXES.items():
                            part["profile"][axis][forms[i]] += counts[key] / whole["n_entries"]
                for i in chosen:
                    part["verdicts"][op["verdict"]][forms[i]] += 1 / len(chosen)

    def shares(part: dict) -> dict:
        return {
            "verdicts": {v: {k: round(n, 6) for k, n in part["verdicts"][v].items() if n} for v in ("positive", "negative", "neutral", "unresolved")},
            "profile": {axis: {k: round(100 * n / part["profiled"], 8) for k, n in part["profile"][axis].items() if n} for axis in AXES},
        }

    for row in dataset["rows"]:
        if row["id"] not in acc:
            continue
        a, system = acc[row["id"]], systems[row["id"]]
        assert a["profiled"] == row["profiled_spans"], (row["id"], a["profiled"])
        assert row["task_window"] == TASK_WINDOW, row["id"]
        for run in row["run_metrics"]:
            within: dict[str, dict] = defaultdict(lambda: {"spans": 0, "sum": 0.0})
            for index, window in enumerate(run["windows"]):
                part = a["runs"][run["run"]][index]
                for key, values in part["within_form"].items():
                    within[key]["spans"] += values["spans"]
                    within[key]["sum"] += values["sum"]
                assert part["profiled"] == window["profiled_spans"], (row["id"], run["run"], index)
                window["forms"] = shares(part)
                for verdict, count in window["verdicts"].items():
                    assert abs(sum(window["forms"]["verdicts"][verdict].values()) - count) < 1e-3, (row["id"], run["run"], index, verdict)
                for axis, value in window["profile"].items():
                    if value is not None:
                        assert abs(sum(window["forms"]["profile"][axis].values()) - value) < 1e-6, (row["id"], run["run"], index, axis)
            run["form_profiles"] = {key: {"profiled_spans": values["spans"],
                                          "benefit": 100 * values["sum"] / values["spans"]}
                                    for key, values in within.items()}
        if system == "memorax":
            # The supplementary row's whole-run forms are curated; only its windows are derived.
            whole = shares(a)
            for kind in ("verdicts", "profile"):
                for key, split in row["forms"][kind].items():
                    assert all(abs(whole[kind][key].get(k, 0) - n) < 1e-6 for k, n in split.items()), (row["id"], key)
            row["forms"].pop("windows", None)
            continue
        used = {f for f, n in a["tokens"].items() if n} | {f for v in a["verdicts"].values() for f, n in v.items() if n}
        catalog = [f for f in (*FORMS[system], COPY, OTHER) if f[0] in used]
        tokens = sum(a["tokens"].values())
        row["forms"] = {
            "items": [{"key": k, "label": label, "route": route} for k, label, route in catalog],
            **shares(a),
            "delivered_tokens": {k: round(100 * n / tokens, 6) for k, n in a["tokens"].items() if n},
        }
        for verdict, count in row["verdicts"].items():
            assert abs(sum(row["forms"]["verdicts"][verdict].values()) - count) < 1e-3, (row["id"], verdict)
        for axis, value in row["profile"].items():
            assert abs(sum(row["forms"]["profile"][axis].values()) - value) < 1e-6, (row["id"], axis)
        print(row["id"], "unclassified judged spans:", a["unclassified"], "tokens:", round(tokens))
    atomic_write(RESULTS, json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
