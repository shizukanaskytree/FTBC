#!/usr/bin/env python3
"""Minimal MCP stdio client for the project's official claude-review bridge.

Launches the same server.py registered for Codex (user-level ~/.codex/config.toml),
with the same CLAUDE_BIN/model/effort/timeout env and the project cwd, in a clean
environment (no desktop-session CLAUDE_* variables). Calls one tool, prints the
JSON payload, and exits. review_start workers are detached by the server itself.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

SERVER = '/data7/projects/Auto-claude-code-research-in-sleep/mcp-servers/claude-review/server.py'
PY = '/home/wxf/miniconda3/bin/python3'
CWD = '/data3/snn_prjs/ftbc_neural_networks'
ENV = {
    'HOME': os.environ['HOME'], 'USER': os.environ.get('USER', 'wxf'), 'LANG': 'en_US.UTF-8',
    'PATH': '/home/wxf/.local/bin:/home/wxf/.nvm/versions/node/v22.22.0/bin:/usr/local/bin:/usr/bin:/bin',
    'CLAUDE_BIN': '/data7/projects/Auto-claude-code-research-in-sleep/.aris/claude-review/claude-official.sh',
    'CLAUDE_REVIEW_MODEL': 'claude-opus-5-5', 'CLAUDE_CODE_EFFORT_LEVEL': 'xhigh', 'CLAUDE_REVIEW_TIMEOUT_SEC': '0',
}

def call(tool, arguments):
    p = subprocess.Popen([PY, SERVER], cwd=CWD, env=ENV, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def send(obj):
        p.stdin.write((json.dumps(obj, ensure_ascii=False) + '\n').encode()); p.stdin.flush()
    def recv():
        line = p.stdout.readline()
        if not line:
            raise RuntimeError('server closed: ' + p.stderr.read().decode(errors='replace'))
        return json.loads(line)
    try:
        send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'ftbc-e12-claude-client', 'version': '1'}}})
        init = recv(); assert init['result']['serverInfo']['name'] == 'claude-review', init
        send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': tool, 'arguments': arguments}})
        resp = recv()
    finally:
        p.stdin.close(); p.wait(timeout=60)
    return resp

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tool', choices=['review_start', 'review_reply_start', 'review_status'])
    ap.add_argument('--prompt-file'); ap.add_argument('--thread'); ap.add_argument('--job')
    ap.add_argument('--tools', default='Read,Grep,Glob'); ap.add_argument('--wait', type=int, default=0)
    ap.add_argument('--save', help='write the raw MCP response JSON here (must not exist)')
    a = ap.parse_args()
    if a.tool == 'review_status':
        args = {'jobId': a.job, 'waitSeconds': a.wait}
    else:
        args = {'prompt': Path(a.prompt_file).read_text(), 'tools': a.tools}
        if a.tool == 'review_reply_start':
            args['threadId'] = a.thread
    resp = call(a.tool, args)
    if a.save:
        out = Path(a.save); assert not out.exists(), 'refusing to overwrite ' + str(out)
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(resp, ensure_ascii=False, indent=2) + '\n')
    content = resp['result']['content'][0]['text']
    print(content)
    sys.exit(1 if resp['result'].get('isError') else 0)
