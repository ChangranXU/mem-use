import json
from urllib.parse import quote

import pytest

from injudge.__main__ import _run_manifest
from injudge.agent import AgentConfig
from injudge.report import build_artifact, build_summary, render_summary
from test_injudge import _op, _signals


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("gateway_secret", [None, "other-gateway+secret/456"])
def test_run_manifest_omits_endpoint_paths(encoded, gateway_secret):
    key = "sk-path+secret/123"
    secret = gateway_secret or key
    spelling = quote(secret, safe="") if encoded else secret
    config = AgentConfig(base_url=f"https://gateway.invalid/{spelling}/v1", api_key=key, model="judge")
    manifest = _run_manifest(config, 4)

    assert manifest["base_url"] == "https://gateway.invalid"
    assert key not in json.dumps(manifest) and secret not in json.dumps(manifest) and spelling not in json.dumps(manifest)
    assert config.base_url == f"https://gateway.invalid/{spelling}/v1"


def test_summary_keeps_untrusted_markdown_literal():
    payload = "![image](https://example.invalid/image) [link](https://example.invalid/link) <img src='x'> &copy;"
    row = _op("T1:S1", "negative")
    row.update(n_injections=1, analysis=payload)
    trajectory = {"id": "T1", "episodes": [{
        "execution_identity": {"episode_id": payload}, "span_ids": ["T1:S1"],
    }]}
    artifact = build_artifact({"judge_model": payload}, [trajectory], [row], [], [{"level": "warn", "msg": payload}])
    original = json.dumps(artifact)
    text = render_summary(artifact)

    assert r"\!\[image\]\(https\:\/\/example\.invalid\/image\)" in text
    assert r"\[link\]\(https\:\/\/example\.invalid\/link\)" in text
    assert "&lt;img src='x'&gt; &amp;copy;" in text
    assert "![image]" not in text and "[link](" not in text and "<img" not in text
    assert text.count(r"\!\[image\]") == 4
    assert "[summary.json](summary.json)" in text
    assert json.dumps(artifact) == original


def test_summary_counts_shared_memory_groups_not_steps():
    rows = [
        {**_op(f"T1:S{i}", "positive",
               memories=[{"id": "m1", "namespace": "ns", "item_id": "a", "key": "a"}],
               rubric={"m1": {**_signals(inf="acted_on", pe="improved"), "anchors": [3, 4]}}),
         "n_injections": 1}
        for i in (1, 2)
    ]
    artifact = build_artifact({}, [{"id": "T1", "run_id": "r"}], rows, [], [])

    assert build_summary(artifact)["overall"]["rubric"]["shared_action_groups"] == 1
