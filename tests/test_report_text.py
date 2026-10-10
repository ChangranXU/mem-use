import json
from urllib.parse import quote

import pytest

from injudge.__main__ import _run_manifest
from injudge.agent import AgentConfig
from injudge.report import build_artifact, render_summary
from test_injudge import _op


@pytest.mark.parametrize("encoded", [False, True])
def test_run_manifest_masks_a_known_key_in_the_endpoint_path(encoded):
    key = "sk-path+secret/123"
    spelling = quote(key, safe="") if encoded else key
    config = AgentConfig(base_url=f"https://gateway.invalid/{spelling}/v1", api_key=key, model="judge")
    manifest = _run_manifest(config, 4)

    assert manifest["base_url"] == "https://gateway.invalid/***/v1"
    assert key not in json.dumps(manifest) and spelling not in json.dumps(manifest)
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
