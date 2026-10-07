"""Tiny OpenAI-compatible server used to test the AG2 tool-calling loop without a real LLM.
Behaviour: on the first turn it asks for a tool call (the first offered tool, which must take no arguments); once a tool result is present it answers."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append(body)
        msgs = body["messages"]
        has_tool_result = any(m.get("role") == "tool" for m in msgs)
        is_auditor = "auditor" in (msgs[0].get("content") or "").lower()
        if body.get("tools") and not has_tool_result:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": body["tools"][0]["function"]["name"], "arguments": "{}"}}]}
            finish = "tool_calls"
        else:
            tool_txt = next((m["content"] for m in reversed(msgs) if m.get("role") == "tool"), "")
            text = ("VERDICT: VERIFIED\nAll numbers match the tool output.\nTERMINATE" if is_auditor
                    else f"Fact: tool returned {len(tool_txt)} characters.\nWhat this means: mock explanation.\nTERMINATE")
            msg, finish = {"role": "assistant", "content": text}, "stop"
        out = {"id": "mock", "object": "chat.completion", "created": 0, "model": body.get("model", "mock"),
               "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start(port: int = 0):
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
