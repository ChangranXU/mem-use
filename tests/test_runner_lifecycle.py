from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.annotated_fixtures import AnnotatedBuilder, cure_ref


_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (_ROOT / "agent/node_modules/@earendil-works/pi-ai").is_dir(),
    reason="node or agent runtime packages not installed",
)


def _until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), 'condition did not complete'


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def endpoint():
    release = threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if not release.wait(10):
                return
            messages = body['messages']
            if not any(message['role'] == 'tool' for message in messages):
                opening = next(message['content'] for message in messages if message['role'] == 'user')
                if isinstance(opening, list):
                    opening = '\n'.join(block.get('text', '') for block in opening)
                paths = []
                for line in opening.splitlines():
                    if line.startswith('Always: '):
                        paths = line[len('Always: '):].rstrip('.').split(', ')
                delta = {'tool_calls': [{'index': 0, 'id': 'read', 'type': 'function',
                         'function': {'name': 'read_evidence', 'arguments': json.dumps({'paths': paths})}}]}
                finish = 'tool_calls'
            else:
                answer = {'confidence': 0.5, 'memories': [{'id': 'm1', 'anchors': [],
                    'relevance': {'value': True}, 'accuracy': {'value': 'uncontradicted'},
                    'novelty': {'value': None}, 'influence': {'value': None},
                    'positive_effect': {'value': None}}]}
                delta, finish = {'content': json.dumps(answer)}, 'stop'
            chunk = {'id': 'local', 'object': 'chat.completion.chunk', 'created': 0,
                     'model': 'test-model', 'choices': [{'index': 0, 'delta': delta,
                     'finish_reason': None}]}
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                chunk['choices'] = [{'index': 0, 'delta': {}, 'finish_reason': finish}]
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', release, requests
    release.set()
    server.shutdown()
    server.server_close()
    worker.join(5)


def _cli(tmp_path, endpoint, parallel):
    builder = AnnotatedBuilder()
    for index in range(4):
        name = f'episode-{index}'
        builder.episode_open(name).agent_turn(with_task=True)
        builder.recall(cure_ref(name, f'fact {index}')).episode_close(name)
    source = builder.write(tmp_path / 'input')
    env = tmp_path / '.env'
    env.write_text(f'BASE_URL={endpoint}\nAPI_KEY=test-key-only\nMODEL=test-model\n'
                   f'JUDGE_PARALLEL={parallel}\nTIMEOUT=10\nMAX_OUTPUT_TOKENS=1000\nMAX_ATTEMPTS=1\n')
    return subprocess.Popen([sys.executable, '-m', 'injudge', str(source), '--env', str(env),
                             '-o', str(tmp_path / 'output')], cwd=_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)


def _report(tmp_path):
    reports = list((tmp_path / 'output').glob('*/injudge-report.json'))
    assert len(reports) == 1
    return json.loads(reports[0].read_text())


@pytest.mark.parametrize('parallel', [1, 2])
def test_terminal_interrupt_retains_running_spans_and_cancels_queue(tmp_path, endpoint, parallel):
    url, release, requests = endpoint
    proc = _cli(tmp_path, url, parallel)
    try:
        _until(lambda: len(requests) == parallel)
        os.killpg(proc.pid, signal.SIGINT)
        time.sleep(0.1)
        release.set()
        stdout, stderr = proc.communicate(timeout=12)
        assert proc.returncode != 0, (stdout, stderr)
        verdicts = [row['verdict'] for row in _report(tmp_path)['operations']]
        assert verdicts.count('unresolved') == parallel, (verdicts, stderr.decode())
        assert verdicts.count('unjudged') == 4 - parallel
        assert len(requests) == 2 * parallel
        assert not list((tmp_path / 'output').rglob('.runner-*'))
    finally:
        release.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


def test_second_terminal_interrupt_stops_running_children(tmp_path, endpoint):
    url, release, requests = endpoint
    proc = _cli(tmp_path, url, 2)
    try:
        _until(lambda: len(requests) == 2)
        os.killpg(proc.pid, signal.SIGINT)
        time.sleep(0.1)
        os.killpg(proc.pid, signal.SIGINT)
        proc.communicate(timeout=8)
        assert proc.returncode != 0
        assert {row['verdict'] for row in _report(tmp_path)['operations']} == {'unjudged'}
        assert len(requests) == 2
        assert not list((tmp_path / 'output').rglob('.runner-*'))
    finally:
        release.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


@pytest.mark.parametrize('termination', [signal.SIGTERM, signal.SIGKILL])
def test_parent_death_stops_runner_and_removes_temporary_files(tmp_path, endpoint, termination):
    url, release, requests = endpoint
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'manifest.json').write_text(json.dumps({'required_reads': {'always': []}}))
    child_pid = tmp_path / 'node.pid'
    script = '''
import sys
from pathlib import Path
import injudge.agent as agent
root, url = Path(sys.argv[1]), sys.argv[2]
original = agent.subprocess.Popen
def launch(*args, **kwargs):
    proc = original(*args, **kwargs)
    (root / 'node.pid').write_text(str(proc.pid))
    return proc
agent.subprocess.Popen = launch
cfg = agent.AgentConfig(base_url=url, api_key='test-key-only', model='test-model',
                        max_output_tokens=1000, timeout=10)
agent._invoke_runner(cfg, root / 'evidence', root / 'trace.jsonl', 1,
                     'test', 'test', None, [], 4)
'''
    proc = subprocess.Popen([sys.executable, '-c', script, str(tmp_path), url], cwd=_ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    pid = None
    try:
        _until(lambda: bool(requests) and child_pid.exists())
        pid = int(child_pid.read_text())
        assert _alive(pid)
        assert list(tmp_path.glob('.runner-*'))
        os.kill(proc.pid, termination)
        proc.communicate(timeout=5)
        _until(lambda: not _alive(pid))
        assert not list(tmp_path.glob('.runner-*'))
        assert len(requests) == 1
    finally:
        release.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)
        if pid is not None and _alive(pid):
            os.kill(pid, signal.SIGKILL)
