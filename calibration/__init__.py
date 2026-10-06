"""Controlled judge-calibration cases and report scoring.

Parser tests with canned answers cannot show that the judge's labels are
right. These development regressions run through the REAL judge: each is a
synthetic schema-v10 episode delivering one memory (a `followup` case adds the
recorded full read its preview selected, scored as a second target,
`<case>/read`). They cover explicit and
uncited use, routine practice, source and novelty distinctions, uncertainty,
plan execution and reversal, harmful advice, unproductive use, reading a
listed candidate whose unrecorded full text is then set aside or used,
recorded preview/full-read pairs, a historically scoped claim, and
paraphrased baselines.
Only labels decided by the construction are pinned.

`score` compares injudge reports against those expectations. It never picks
a preferred answer: every disagreement is listed, repeated reports of the
same cases give per-label stability, and the matched pair's novelty is
compared within each report.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from tests.annotated_fixtures import AnnotatedBuilder, cure_ref

_SIGNALS = ("relevance", "accuracy", "novelty", "influence", "positive_effect")

# Each case: task text, turns before the delivery, the memory, the consumer
# turn (the first to see the memory), later turns, and the acceptable
# values of each label it pins. Turns are (reasoning, command, result).
CASES: dict[str, dict] = {
    "explicit_use": {
        "task": "tests/unit/test_parser.py fails with a cache permission error "
                "before any test runs. Fix the parser so test_parse_empty passes.",
        "before": [
            ("Run the failing test to see the error.",
             "pytest tests/unit/test_parser.py",
             "ERROR: [Errno 13] Permission denied: '.pytest_cache'"),
        ],
        "memory": ("pytest_readonly_cache",
                   "In this sandbox the .pytest_cache directory is read-only; run "
                   "pytest with -p no:cacheprovider to get real test results."),
        "consumer": ("The memory says the cache is read-only here and to pass "
                     "-p no:cacheprovider. Rerun that way.",
                     "pytest -p no:cacheprovider tests/unit/test_parser.py",
                     "FAILED test_parse_empty - IndexError: list index out of range\n"
                     "1 failed, 7 passed"),
        "after": [
            ("The real failure is an IndexError on empty input. Guard it.",
             "sed -i 's/return tokens\\[0\\]/return tokens[0] if tokens else None/' src/parser.py",
             ""),
            ("Rerun the tests.",
             "pytest -p no:cacheprovider tests/unit/test_parser.py", "8 passed"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "implicit_action": {
        "task": "Setting APP_TIMEOUT in the environment has no effect on the client "
                "timeout. Make the environment variable override the default.",
        "before": [
            ("Find where the timeout default is defined.",
             "grep -rn TIMEOUT src/app/config.py", "3:DEFAULT_TIMEOUT = 30"),
            ("config.py only defines the constant; find its users.",
             "grep -rn 'DEFAULT_TIMEOUT\\|settings.timeout' src/app",
             "src/app/config.py:3:DEFAULT_TIMEOUT = 30\n"
             "src/app/client.py:12:    timeout = settings.timeout"),
        ],
        "memory": ("settings_env_overrides",
                   "Environment overrides for app settings are applied in "
                   "src/app/settings/_env.py (function apply_env), not in config.py."),
        "consumer": ("Open the override module.",
                     "sed -n 1,40p src/app/settings/_env.py",
                     "def apply_env(settings):\n"
                     "    for name in ('APP_HOST', 'APP_PORT'):\n"
                     "        setattr(settings, name[4:].lower(), os.environ.get(name))"),
        "after": [
            ("APP_TIMEOUT is missing from the override loop; add it with int conversion.",
             "python edit_env.py src/app/settings/_env.py", "edited"),
            ("Verify.",
             "APP_TIMEOUT=5 python -c 'from app.client import make; print(make().timeout)'",
             "5"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "redundant_instruction": {
        "task": "Fix the off-by-one in paginate() so the last page is returned. "
                "Run the suite with `make test` before finishing; do not edit tests.",
        "before": [
            ("Read paginate().", "sed -n 1,30p src/pages.py",
             "def paginate(items, size):\n"
             "    return [items[i:i+size] for i in range(0, len(items) - size, size)]"),
            ("The range stops one page early. Fix the bound, then run make test "
             "as the task says.",
             "sed -i 's/len(items) - size/len(items)/' src/pages.py", ""),
        ],
        "memory": ("run_tests", "Run the test suite with make test and do not modify test files."),
        "consumer": ("Now run the suite.", "make test", "12 passed"),
        "after": [("All tests pass.", "git diff --stat", " src/pages.py | 2 +-")],
        "expect": {"relevance": [True], "novelty": ["redundant"], "influence": ["none"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "post_delivery_plan": {
        "task": "Importing the package fails in CI with 'numpy.core.multiarray "
                "failed to import'. Make the import work.",
        "before": [
            ("Reproduce the import failure.", "python -c 'import pkg'",
             "ImportError: numpy.core.multiarray failed to import"),
        ],
        "memory": ("numpy_abi_pin",
                   "This project's compiled extension is built against NumPy 1.x; "
                   "pin numpy<2 in requirements.txt to fix the import error."),
        "consumer": ("I was already planning to pin numpy below 2.",
                     "sed -i 's/^numpy$/numpy<2/' requirements.txt && pip install -r requirements.txt",
                     "Successfully installed numpy-1.26.4"),
        "after": [("Check the import.", "python -c 'import pkg; print(pkg.__version__)'", "0.4.1")],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "mixed_content": {
        "task": "tests/test_schedule.py::test_next_run is flaky. Make it pass "
                "reliably. Run tests with `make test`.",
        "before": [
            ("Run the suite.", "make test",
             "FAILED tests/test_schedule.py::test_next_run - AssertionError: "
             "'10:00' != '09:00'\n1 failed, 30 passed"),
            ("A one-hour mismatch. Inspect the test.", "sed -n 1,20p tests/test_schedule.py",
             "def test_next_run():\n    assert next_run('09:00').strftime('%H:%M') == '09:00'"),
        ],
        "memory": ("schedule_tests",
                   "Run tests with make test. test_next_run depends on the local "
                   "timezone because next_run calls datetime.now() without tzinfo; "
                   "use datetime.now(timezone.utc) in src/schedule.py."),
        "consumer": ("Make next_run use UTC explicitly.",
                     "sed -i 's/datetime.now()/datetime.now(timezone.utc)/' src/schedule.py", ""),
        "after": [("Rerun.", "make test", "31 passed")],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "mixed_irrelevant_content": {
        "task": "Fix encode_id(0) returning an empty string. Run make test to verify.",
        "before": [
            ("Inspect encode_id.", "cat src/ids.py",
             "def encode_id(value):\n    return str(value) if value else ''"),
            ("Zero should be encoded as '0'; only None is empty. Fix that, then run make test.",
             "python replace_falsy_check.py src/ids.py", "now checks value is not None"),
        ],
        "memory": ("prior_transfer_work",
                   "Run make test to verify the repair. In a separate file-transfer service, "
                   "protocol revision 5 uses 64-KiB frames and a two-byte checksum."),
        "consumer": ("Run the planned verification.", "make test", "22 passed"),
        "after": [],
        "expect": {"novelty": ["novel"], "influence": ["none"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "harmful_advice": {
        "task": "`python -m tool build` crashes with KeyError: 'output_dir' when "
                "config.toml omits output_dir. Default output_dir to 'dist'.",
        "before": [("Reproduce.", "python -m tool build", "KeyError: 'output_dir'")],
        "memory": ("build_crash_fix",
                   "Build crashes in this tool come from a stale cache; delete the "
                   ".cache directory and re-run the build."),
        "consumer": ("Following the stored fix, clear the cache first.",
                     "rm -rf .cache && python -m tool build",
                     "error: .cache/index missing; run `python -m tool index` first"),
        "after": [
            ("Deleting the cache removed the prebuilt index. Rebuild it.",
             "python -m tool index", "indexed 18204 files in 21m04s"),
            ("The crash is the missing key. Read the loader.",
             "grep -n output_dir src/tool/config.py", "14:    out = cfg['output_dir']"),
            ("Default it.",
             "sed -i \"s/cfg\\['output_dir'\\]/cfg.get('output_dir', 'dist')/\" src/tool/config.py", ""),
            ("Verify.", "python -m tool build", "built 3 targets into dist/"),
        ],
        "expect": {"relevance": [True], "accuracy": ["contradicted"],
                   "influence": ["acted_on"],
                   "positive_effect": ["harmed"], "verdict": ["negative"]},
    },
    "unrelated_mechanism": {
        "task": "Dataset.copy(deep=True) turns a '<U3' string index into object "
                "dtype. Keep the index dtype when copying.",
        "before": [
            ("Reproduce.", "python repro.py", "before: <U3  after: object"),
        ],
        "memory": ("setitem_regression",
                   "In 0.10.1 Variable.__setitem__ unwrapped a scalar object with a "
                   ".values attribute on assignment; wrapping the scalar in a 0-d "
                   "object array before as_compatible_data preserved it."),
        "consumer": ("Look at how copy rebuilds the index adapter.",
                     "grep -n 'def copy' -A 12 xarray/core/variable.py",
                     "    def copy(self, deep=True, data=None):\n"
                     "        ...\n"
                     "        data = PandasIndexAdapter(self._data.array.copy(deep=True))"),
        "after": [
            ("The adapter drops the dtype. Pass it through.",
             "sed -i 's/copy(deep=True))/copy(deep=True), dtype=self.dtype)/' xarray/core/variable.py",
             ""),
            ("Verify.", "python repro.py", "before: <U3  after: <U3"),
        ],
        "expect": {"relevance": [False], "influence": ["none"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "generic_practice": {
        "task": "slugify('Ünïcode') returns an empty string. Fix it and verify "
                "your fix.",
        "before": [
            ("Reproduce.", "python -c \"from text import slugify; print(repr(slugify('Ünïcode')))\"",
             "''"),
        ],
        "memory": ("validation_order",
                   "Validate a fix by running the focused test module first, then the "
                   "full suite, and keep the change as small as possible."),
        "consumer": ("slugify drops non-ASCII characters instead of transliterating "
                     "them. Normalize with NFKD before the ASCII encode.",
                     "sed -i 's/value.encode(/unicodedata.normalize(\"NFKD\", value).encode(/' src/text.py",
                     ""),
        "after": [
            ("Run the slugify tests.", "pytest tests/test_text.py -q", "6 passed"),
            ("Run the full suite.", "pytest -q", "212 passed"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["none"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "open_hypothesis": {
        "task": "Identify the running service's active storage backend. Do not change it.",
        "before": [
            ("Inspect the service status.", "service status", "storage_backend=sqlite"),
            ("That field might be a default rather than the active backend. "
             "I am still unsure whether this instance uses SQLite. Check startup logs, "
             "then finish with the already planned health check.",
             "service logs --startup", "startup complete"),
        ],
        "memory": ("active_backend", "The running service uses SQLite as its active storage backend."),
        "consumer": ("Finish the planned health check.",
                     "service health", "healthy"),
        "after": [],
        "expect": {"novelty": ["confirming"]},
    },
    "unexecuted_plan": {
        "task": "Repair the application's failing import.",
        "before": [("Reproduce the failure.", "python -c 'import app'", "ImportError: resolver unavailable")],
        "memory": ("resolver_config", "The resolver is configured in src/runtime/resolver.toml; enable the local backend there."),
        "consumer": ("The memory points to the resolver configuration. I plan to enable "
                     "the local backend, but first check workspace access.",
                     "git status --short", "fatal: workspace no longer exists"),
        "after": [("The workspace is unavailable; stop without opening or changing the configuration.",
                   "pwd", "working directory unavailable")],
        "expect": {"novelty": ["novel"], "influence": ["referenced"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "deferred_use": {
        "task": "Enable the resolver so the application can start. Modify no other settings.",
        "before": [("Reproduce startup.", "app start", "resolver backend not configured")],
        "memory": ("resolver_config", "Set backend='local' in src/runtime/resolver.toml to enable the resolver."),
        "consumer": ("Use the memory's local-backend setting in that file. First confirm "
                     "the repository is clean before applying the chosen change.",
                     "git status --short", ""),
        "after": [
            ("Apply the selected setting.", "sed -i \"s/backend=''/backend='local'/\" src/runtime/resolver.toml", ""),
            ("Verify startup.", "app start", "application ready; local resolver active"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "uncited_abandoned_setting": {
        "task": "Make the image worker start successfully.",
        "before": [("Inspect the failure.", "worker start", "unsupported index format")],
        "memory": ("worker_index", "Set INDEX_READER=v3 in /srv/image/worker.env to load the current index."),
        "consumer": ("I will set INDEX_READER=v3 in /srv/image/worker.env after checking workspace access.",
                     "git status --short", "fatal: workspace no longer exists"),
        "after": [("The workspace is gone. Stop without editing or restarting anything.",
                   "pwd", "working directory unavailable")],
        # A probe of an unexecuted plan must not perform even a preliminary
        # command on the distinctive path supplied only by memory.
        "expect": {"novelty": ["novel"], "influence": ["none", "referenced"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "transferable_technique": {
        "task": "Our httpx-based client hangs forever in tests/test_client.py when "
                "the server stalls. Make fetch fail instead of hanging.",
        "before": [
            ("Look at fetch.", "sed -n 1,20p src/client.py",
             "def fetch(url):\n    with httpx.Client(timeout=None) as c:\n"
             "        return c.get(url)"),
            ("timeout=None disables httpx timeouts. I will pass an explicit timeout.",
             "grep -rn timeout src/", "src/client.py:2:    with httpx.Client(timeout=None) as c:"),
        ],
        "memory": ("requests_hang",
                   "In a requests-based service, hanging tests were fixed by passing "
                   "an explicit timeout to every session call instead of relying on "
                   "the library default."),
        "consumer": ("Set a 10 second timeout.",
                     "sed -i 's/timeout=None/timeout=10.0/' src/client.py", ""),
        "after": [("Run the test.", "pytest tests/test_client.py", "1 passed")],
        "expect": {"relevance": [True], "novelty": ["redundant"], "influence": ["none"],
                   "positive_effect": [None], "verdict": ["neutral"]},
    },
    "routing_uncertainty": {
        "task": "Identify the active region of the upload gateway without changing its configuration.",
        "before": [
            ("Read gateway metadata.", "gateway metadata", "region=eu-north"),
            ("This could be the configured fallback. I still do not know whether "
             "live uploads use eu-north. Inspect the connection, then record uptime.",
             "gateway connection", "connected; region field unavailable"),
        ],
        "memory": ("upload_region", "Live uploads through this gateway are routed to eu-north."),
        "consumer": ("Record uptime.", "gateway uptime", "18 hours"),
        "after": [],
        "expect": {"novelty": ["confirming"]},
    },
    "uncited_deferred_flag": {
        "task": "Start the catalog service within this sandbox's memory limit.",
        "before": [("Reproduce startup.", "catalogd start", "eager catalog allocation: out of memory")],
        "memory": ("catalog_startup", "catalogd supports --catalog-mode lazy; it loads entries on demand and avoids the startup allocation."),
        "consumer": ("Start with --catalog-mode lazy after checking for an existing process.",
                     "pgrep catalogd", "no matching process"),
        "after": [
            ("Start the chosen configuration.", "catalogd start --catalog-mode lazy",
             "ready; 48 MB resident; catalog entries loaded on demand"),
            ("Exercise a catalog lookup.", "catalogctl lookup sample", "sample found"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    "routine_reversal": {
        "task": "Fix parse_header accepting an empty header name and verify the repair.",
        "before": [
            ("Inspect test locations.", "ls tests", "test_headers.py test_body.py integration/"),
            ("I always begin with the full integration suite. I will run that next, "
             "before any focused test, even though it takes two hours.",
             "git status --short", ""),
        ],
        "memory": ("test_order", "Run the relevant focused test before the full integration suite; use its failure to localize the repair."),
        "consumer": ("Change the order: run the header test first and use its failure to locate the defect.",
                     "pytest tests/test_headers.py -q",
                     "FAILED test_empty_name: src/headers.py:18 accepted empty name; 0.2 seconds"),
        "after": [
            ("The focused failure locates the missing empty-name check at line 18. Add it.",
             "python add_empty_name_guard.py src/headers.py", "guard added at line 18"),
            ("Check the repaired case, then integration.",
             "pytest tests/test_headers.py -q && pytest tests/integration -q",
             "12 passed; 340 passed"),
        ],
        "expect": {"novelty": ["novel"], "influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    # The full text comes back as plain tool output, not a recorded memory
    # delivery, so no other span judges it: the read is this entry's action
    # and its consequence this entry's effect. The plan is voiced before the
    # read and the read's advice shares nothing with the fix.
    "selected_then_rejected": {
        "task": "cache.get() returns a stale value right after cache.set() on the "
                "same key. Fix it.",
        "before": [
            ("Reproduce.", "python repro.py", "set 2 -> get 1 (stale)"),
            ("set() probably leaves the per-key memo stale. Check set() next.",
             "grep -n 'def set' src/cache.py", "14:    def set(self, k, v):"),
        ],
        "memory": ("kb-4821", "Issue: stale reads in an in-memory cache "
                              "(read the full fix with: kb read kb-4821)"),
        "consumer": ("Candidate kb-4821 looks related; read it.",
                     "kb read kb-4821",
                     "kb-4821 fix: give every cache entry a one-second TTL so stale "
                     "values expire."),
        "after": [
            ("A TTL would only hide the bug. Continue with set() as planned.",
             "sed -n 14,17p src/cache.py",
             "def set(self, k, v):\n    self._store[k] = v\n    # self._memo is never cleared"),
            ("set() leaves the per-key memo stale. Drop the key from _memo in set().",
             "sed -i 's/self._store\\[k\\] = v/self._store[k] = v; self._memo.pop(k, None)/' src/cache.py",
             ""),
            ("Verify.", "python repro.py", "set 2 -> get 2"),
        ],
        "expect": {"influence": ["acted_on"],
                   "positive_effect": ["no_effect"], "verdict": ["neutral"]},
    },
    # As above, but the unrecorded read supplies the fix.
    "selected_then_used": {
        "task": "Find why config resolution ignores the project-local overrides and fix it.",
        "before": [("Reproduce.", "python config_repro.py", "project-local override ignored")],
        "memory": ("kb-927", "Candidate kb-927: project-local override resolution. "
                             "Read kb-927 for the fix."),
        "consumer": ("Read candidate kb-927.", "kb read kb-927",
                     "Resolver checks defaults first. Reverse the search order in src/resolver.py."),
        "after": [
            ("The read says to check project-local values first; apply it.",
             "python reverse_resolver_order.py", "search order reversed"),
            ("Verify.", "python config_repro.py", "project-local override applied"),
        ],
        "expect": {"influence": ["acted_on"],
                   "positive_effect": ["improved"], "verdict": ["positive"]},
    },
    # Acted on without benefit or harm: unproductive, which is not misuse.
    "unproductive_action": {
        "task": "Fix the parser's empty input crash.",
        "before": [("The crash is an unguarded first-token access; add an empty-input guard next.",
                    "cat src/parser.py", "def parse(tokens): return tokens[0]")],
        "memory": ("parser_probe", "Run parser-probe --dump-allocation before changing this parser."),
        "consumer": ("The memory says to dump allocations; try that first.",
                     "parser-probe --dump-allocation", "allocation dump written to probe.bin"),
        "after": [
            ("The dump has no useful information for this bug. Discard it unread and "
             "carry out the guard already planned.",
             "rm probe.bin && python add_empty_input_guard.py", "guard added"),
            ("Verify the guard.", "pytest tests/test_parser.py", "8 passed"),
        ],
        "expect": {"influence": ["acted_on"],
                   "positive_effect": ["no_effect"], "verdict": ["neutral"]},
    },
    # A claim scoped to an old release is not refuted by today's code; the
    # mismatch is a reason it does not apply, not evidence it was false.
    "historical_scope": {
        "task": "Config.load() ignores the YAML file's timeout key and always uses "
                "30. Fix it.",
        "before": [
            ("Find the loader.", "sed -n 1,8p src/config.py",
             "def load(path):\n    with open(path) as fh:\n        data = yaml.safe_load(fh)\n"
             "    return Config(timeout=30, **{k: v for k, v in data.items() if k != 'timeout'})"),
        ],
        "memory": ("loader_handle_leak",
                   "In release 1.2, Config.load() opened the YAML file without closing "
                   "it and leaked file handles; wrapping open() in a with block fixed it."),
        "consumer": ("load() hard-codes timeout=30 and drops the file's value. Pass it "
                     "through with 30 as the default.",
                     "sed -i \"s/timeout=30, \\*\\*{k: v for k, v in data.items() if k != 'timeout'}/"
                     "**{'timeout': 30, **data}/\" src/config.py",
                     ""),
        "after": [
            ("Verify.", "python -c 'from config import load; print(load(\"t.yaml\").timeout)'",
             "5"),
        ],
        "expect": {"relevance": [False], "accuracy": ["uncontradicted", "corroborated"],
                   "influence": ["none"], "positive_effect": [None], "verdict": ["neutral"]},
    },
    "recorded_followup_used": {
        "task": "The parser rejects hexadecimal identifiers. Make parse_id('0xff') return 255.",
        "before": [("Reproduce.", "python reproduce.py", "ValueError: invalid literal '0xff'")],
        "memory": ("kb-hex", "ID kb-hex: parse_id must pass base=0 to int to accept the 0x prefix."),
        "consumer": ("Read the full kb-hex experience before changing the parser.", "kb read kb-hex", ""),
        "followup": {
            "value": "ID kb-hex: parse_id must pass base=0 to int to accept the 0x prefix. Change the conversion in src/identifiers.py and verify decimal identifiers still parse.",
            "expect": {"influence": ["acted_on"], "positive_effect": ["improved"], "verdict": ["positive"]},
        },
        "after": [("The full experience gives the conversion and location. Apply base=0 in src/identifiers.py.",
                   "python set_integer_base.py src/identifiers.py", "conversion updated"),
                  ("Verify both formats.", "python reproduce.py", "0xff -> 255; 123 -> 123")],
        "expect": {"influence": ["selected"], "positive_effect": [None], "verdict": ["neutral"]},
    },
    "recorded_followup_unused": {
        "task": "Fix deletion of the first entry from a queue.",
        "before": [("delete(0) does nothing. The falsy-index condition should distinguish None from zero; fix it next.",
                    "cat src/queue.py", "def delete(i):\n    if i: entries.pop(i)")],
        "memory": ("kb-queue", "ID kb-queue: queue deletions can fail because cached index maps become stale."),
        "consumer": ("Check the full kb-queue experience before continuing.", "kb read kb-queue", ""),
        "followup": {
            "value": "ID kb-queue: queue deletions can fail because cached index maps become stale. Rebuild index_map after every insertion or removal.",
            "expect": {"influence": ["referenced"], "positive_effect": [None], "verdict": ["neutral"]},
        },
        "after": [("The fetched experience is about an index_map, which this queue does not have. Continue the already planned None guard.",
                   "python fix_none_guard.py src/queue.py", "if i is not None: entries.pop(i)"),
                  ("Verify.", "python queue_repro.py", "delete(0) removes first entry")],
        "expect": {"influence": ["selected"], "positive_effect": [None], "verdict": ["neutral"]},
    },
    "recorded_followup_after_action": {
        "task": "Make parse_number('0b11') return 3 instead of raising ValueError.",
        "before": [("Reproduce.", "python binary_repro.py", "ValueError: invalid literal '0b11'")],
        "memory": ("kb-binary", "ID kb-binary: use int(value, 0) in src/numbers.py to recognize binary prefixes."),
        "consumer": ("The preview gives the exact fix and location. Apply it now, then fetch kb-binary for any additional details.",
                     "python set_base_zero.py src/numbers.py", "conversion updated"),
        "followup": {
            "lookup": "The preview fix is applied. Read kb-binary for any additional details.",
            "value": "ID kb-binary: use int(value, 0) in src/numbers.py to recognize binary prefixes. This is the complete fix; the base-zero conversion is all that is needed.",
            "expect": {"influence": ["referenced"], "positive_effect": [None], "verdict": ["neutral"]},
        },
        "after": [("The full read repeats the already applied preview, with no additional change. Run the verification already planned for that edit.",
                   "python binary_repro.py", "0b11 -> 3"),
                  ("Check the patch.", "git diff", "-return int(value)\n+return int(value, 0)")],
        "expect": {"influence": ["acted_on"], "positive_effect": ["improved"], "verdict": ["positive"]},
    },
}

# The matched pair: identical memory and task, baselines that state the same
# substance in different words. Novelty must agree within one report.
_ISO = {
    "task": "parse_date('2024-01-01T00:00:00Z') raises ValueError. Fix date "
            "parsing for ISO strings with a trailing Z.",
    "memory": ("iso_z_suffix",
               "datetime.fromisoformat on Python 3.10 rejects a trailing Z; "
               "replace Z with +00:00 before parsing."),
    "consumer": ("Apply the replacement.",
                 "sed -i \"s/fromisoformat(s)/fromisoformat(s.replace('Z', '+00:00'))/\" src/dates.py",
                 ""),
    "after": [("Test.", "python -c \"from dates import parse_date; "
                        "print(parse_date('2024-01-01T00:00:00Z'))\"",
               "2024-01-01 00:00:00+00:00")],
    "expect": {"novelty": ["redundant"], "influence": ["none"],
               "positive_effect": [None], "verdict": ["neutral"]},
    "pair": "iso_z",
}
_ISO_READ = ("sed -n 1,15p src/dates.py", "def parse_date(s):\n    return datetime.fromisoformat(s)")
CASES["matched_a"] = {**_ISO, "before": [
    ("parse_date calls datetime.fromisoformat, which rejects a trailing 'Z' on "
     "Python 3.10. I'll replace the Z with +00:00 before parsing.", *_ISO_READ)]}
CASES["matched_b"] = {**_ISO, "before": [
    ("Root cause: on Python 3.10 fromisoformat does not accept the 'Z' suffix, so "
     "parse_date must map Z to +00:00 first.", *_ISO_READ)]}


_MEMORY_MSG = 4  # past the builder's task (1), reply (2), and tool-result (3) slots


def case_id(name: str) -> str:
    """The opaque identity a case is recorded under. The judge reads run and
    episode ids, so a descriptive name would hint at the expected labels."""
    return "ep-" + hashlib.sha256(name.encode()).hexdigest()[:8]


def build(root: Path) -> list[Path]:
    """Write one trajectory directory per case, named by its opaque id."""
    out = []
    for name, case in CASES.items():
        ident = case_id(name)
        b = AnnotatedBuilder(task_ref={"availability": "inline", "text": case["task"]},
                             instance_id=ident)
        b.episode_open(ident, treatment_arm_id="arm-a").user_message(case["task"])
        for reasoning, command, result in case["before"]:
            b.agent_turn(reasoning=reasoning, command=command, result=result)
        key, value = case["memory"]
        ref = cure_ref(key, value)
        search = b.search(returned=[ref], query=case["task"])
        # Like a real bridge: the memory rides its own message in the
        # consumer's request, and the delivery carries the proof naming it.
        text = f"- [insight] {key}: {value}"
        b.user_message(text, msg=_MEMORY_MSG)
        reasoning, command, result = case["consumer"]
        followup = case.get("followup")
        if followup and not followup.get("lookup"):
            b.agent_turn(reasoning=reasoning, tool_name="exp_read", tool_args={"id": key}, result=followup["value"])
        else:
            b.agent_turn(reasoning=reasoning, command=command, result=result)
        b.delivery(search, memories=[ref], bind=b.last_call, content=text)
        b.events[-1]["binding"] = {"proofs": [{
            "slot": "memory_block", "kind": "canonical_message", "msg_index": _MEMORY_MSG,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest()}]}
        if followup and followup.get("lookup"):
            b.agent_turn(reasoning=followup["lookup"], tool_name="exp_read", tool_args={"id": key}, result=followup["value"])
        for i, (reasoning, command, result) in enumerate(case["after"]):
            if followup and i == 0:
                full = cure_ref(key, followup["value"])
                search = b.search(returned=[full], query=key, trigger="explicit")
            b.agent_turn(reasoning=reasoning, command=command, result=result)
            if followup and i == 0:
                b.delivery(search, memories=[full], bind=b.last_call, content=followup["value"],
                           placements=[{"slot": "memory_block", "kind": "tool_result",
                                        "proof_kind": "canonical_message", "content": {
                                            "availability": "inline", "text": followup["value"]}}])
                b.events[-1]["binding"] = {"proofs": [{
                    "slot": "memory_block", "kind": "canonical_message", "msg_index": 3,
                    "content_sha256": hashlib.sha256(followup["value"].encode()).hexdigest()}]}
        b.episode_close(ident, treatment_arm_id="arm-a")
        out.append(b.write(root / ident, run_id=ident, task_preview=case["task"]))
    return out


def expected_labels() -> dict[str, dict]:
    """Each recorded delivery is a target, including the follow-up read."""
    out = {}
    for name, case in CASES.items():
        out[name] = case["expect"]
        if case.get("followup"):
            out[name + "/read"] = case["followup"]["expect"]
    return out


def _labels(report: dict) -> dict[str, dict]:
    """Unexpected, missing, or failed spans remain unscored, never stable."""
    names = {case_id(name): name for name in CASES}
    runs = {t["id"]: names.get(t.get("run_id")) for t in report.get("trajectories") or []}
    ops: dict[str, list[dict]] = {}
    for op in report.get("operations") or []:
        if not op.get("auto"):
            ops.setdefault(runs.get(op.get("traj")), []).append(op)
    out: dict[str, dict] = {}
    for target in expected_labels():
        name = target.removesuffix("/read")
        found = ops.get(name, [])
        required = 2 if CASES[name].get("followup") else 1
        if len(found) != required:
            out[target] = {"problem": f"expected {required} spans, found {len(found)}"}
            continue
        op = found[1 if target.endswith("/read") else 0]
        if op.get("error") or "m1" not in (op.get("rubric") or {}):
            out[target] = {"problem": op.get("error") or "no m1 rubric"}
            continue
        entry = op["rubric"]["m1"]
        out[target] = {s: (entry.get(s) or {}).get("value") for s in _SIGNALS}
        out[target]["verdict"] = op.get("verdict")
        out[target]["notes"] = {s: (entry.get(s) or {}).get("note") for s in _SIGNALS}
    return out


def score(reports: list[dict]) -> list[str]:
    """Agreement, disagreements, stability, and matched-pair lines."""
    lines: list[str] = []
    labeled = [_labels(r) for r in reports]
    expected = expected_labels()
    for i, cases in enumerate(labeled, 1):
        agree = total = 0
        misses = []
        for name, labels in expected.items():
            got = cases[name]
            for label, allowed in labels.items():
                total += 1
                if "problem" in got:
                    misses.append(f"    {name}.{label}: unscored ({got['problem']})")
                elif got[label] in allowed:
                    agree += 1
                else:
                    note = got["notes"].get(label) if label in _SIGNALS else None
                    misses.append(f"    {name}.{label}: expected {allowed}, got {got[label]!r}"
                                  + (f" — {note}" if note else ""))
        lines.append(f"report {i}: {agree}/{total} expected labels agree")
        lines.extend(misses)
    if len(labeled) > 1:
        # Only labels every report produced can be stable; a missing or
        # errored case is unavailable, never an agreement.
        stable = total = unavailable = 0
        unstable = []
        for name in expected:
            for label in (*_SIGNALS, "verdict"):
                if any("problem" in c[name] for c in labeled):
                    unavailable += 1
                    continue
                values = [c[name][label] for c in labeled]
                total += 1
                if len(set(map(repr, values))) == 1:
                    stable += 1
                else:
                    counts = Counter(map(repr, values))
                    unstable.append(f"    {name}.{label}: " + ", ".join(
                        f"{v} ×{n}" for v, n in counts.most_common()))
        lines.append(f"stability: {stable}/{total} labels identical across {len(labeled)} reports"
                     + (f"; {unavailable} unavailable (missing or errored in a report)" if unavailable else ""))
        lines.extend(unstable)
    pairs: dict[str, list[str]] = {}
    for name, case in CASES.items():
        if case.get("pair"):
            pairs.setdefault(case["pair"], []).append(name)
    for pair, names in pairs.items():
        for i, cases in enumerate(labeled, 1):
            values = [cases[n].get("novelty", cases[n].get("problem")) for n in names]
            if any("problem" in cases[n] for n in names):
                status = "unavailable"
            else:
                status = "consistent" if len(set(map(repr, values))) == 1 else "INCONSISTENT"
            lines.append(f"matched pair {pair}, report {i}: novelty {status} "
                         + ", ".join(f"{n}={v!r}" for n, v in zip(names, values)))
    return lines
