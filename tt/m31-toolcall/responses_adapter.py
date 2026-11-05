#!/usr/bin/env python3
"""Minimal OpenRouter-style Responses -> Chat Completions adapter for running OpenRouter's
benchmark-harness (toolcall_formats, toolcall_schema_fuzz) directly against a Dynamo frontend.

OpenRouter sends a provider Chat Completions requests and turns the reply into a Responses
result; this does the same, so the harness scores our serving stack, not Dynamo's /v1/responses.
Only what the tool-call suites need: messages, tools, temperature, max tokens; the harness reads
only the terminal `response.completed` event, so the reply is that one SSE event.

  UPSTREAM=http://<frontend>:8000 MODEL=MiniMaxAI/MiniMax-M3.1-Flash STREAM=1 \
  LOG=/path/raw.jsonl PORT=18080 python3 responses_adapter.py
  OPENROUTER_BASE_URL=http://127.0.0.1:18080/api/v1 OPENROUTER_API_KEY=dummy bun run bench -- ...

STREAM=1 calls upstream with stream=true and assembles the deltas (exercises the streaming
parser, as OpenRouter's provider calls do); STREAM=0 calls non-streaming. Stdlib only.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "MiniMaxAI/MiniMax-M3.1-Flash")
STREAM = os.environ.get("STREAM", "1") == "1"
LOG = os.environ.get("LOG")
PORT = int(os.environ.get("PORT", "18080"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "6000"))
TIMEOUT = float(os.environ.get("TIMEOUT", "600"))
_log_lock = threading.Lock()


def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
            parts.append(part.get("text", ""))
    return "".join(parts)


def to_chat(req):
    messages = []
    if req.get("instructions"):
        messages.append({"role": "system", "content": req["instructions"]})
    items = req.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    pending_calls = []
    for item in items or []:
        kind = item.get("type", "message")
        if kind == "message":
            if pending_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            role = item.get("role", "user")
            role = "system" if role == "developer" else role
            messages.append({"role": role, "content": _text_of(item.get("content"))})
        elif kind == "function_call":
            pending_calls.append({
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {"name": item["name"], "arguments": item.get("arguments", "{}")},
            })
        elif kind == "function_call_output":
            if pending_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            output = item.get("output")
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": output if isinstance(output, str) else json.dumps(output),
            })
    if pending_calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})

    tools = []
    for tool in req.get("tools") or []:
        if tool.get("type") != "function":
            continue
        fn = {"name": tool["name"], "parameters": tool.get("parameters") or {"type": "object"}}
        if tool.get("description"):
            fn["description"] = tool["description"]
        if "strict" in tool and tool["strict"] is not None:
            fn["strict"] = tool["strict"]
        tools.append({"type": "function", "function": fn})

    chat = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": req.get("max_output_tokens") or DEFAULT_MAX_TOKENS,
        "stream": STREAM,
    }
    if tools:
        chat["tools"] = tools
    if req.get("tool_choice") not in (None, "auto"):
        chat["tool_choice"] = req["tool_choice"]
    for key in ("temperature", "top_p"):
        if req.get(key) is not None:
            chat[key] = req[key]
    if STREAM:
        chat["stream_options"] = {"include_usage": True}
    return chat


def call_upstream(chat):
    body = json.dumps(chat).encode()
    request = urllib.request.Request(
        UPSTREAM + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
        if not STREAM:
            data = json.loads(resp.read())
            choice = data["choices"][0]
            msg = choice["message"]
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "argument_chunks": None,
            }
        content, reasoning, calls, finish, usage = [], [], {}, None, None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": [], "n": 0})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])
                        slot["n"] += 1
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        tool_calls = [
            {"id": s["id"] or f"call_{i}", "type": "function",
             "function": {"name": s["name"], "arguments": "".join(s["args"])}}
            for i, s in sorted(calls.items())
        ]
        return {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_calls": tool_calls,
            "finish_reason": finish,
            "usage": usage,
            "argument_chunks": [s["n"] for _, s in sorted(calls.items())],
        }


def to_response(req, result):
    output = []
    if result["reasoning"]:
        output.append({
            "type": "reasoning", "id": "rs_" + uuid.uuid4().hex[:12], "summary": [],
            "content": [{"type": "reasoning_text", "text": result["reasoning"]}],
        })
    if result["content"]:
        output.append({
            "type": "message", "id": "msg_" + uuid.uuid4().hex[:12], "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": result["content"], "annotations": []}],
        })
    for tc in result["tool_calls"]:
        fn = tc.get("function") or {}
        output.append({
            "type": "function_call", "id": "fc_" + uuid.uuid4().hex[:12],
            "call_id": tc.get("id") or "call_" + uuid.uuid4().hex[:8],
            "name": fn.get("name", ""), "arguments": fn.get("arguments") or "",
            "status": "completed",
        })
    usage = result.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    status = "incomplete" if result.get("finish_reason") == "length" else "completed"
    return {
        "id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()),
        "model": MODEL, "status": status, "output": output,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "tools": req.get("tools") or [], "temperature": req.get("temperature"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": details.get("reasoning_tokens") or 0},
        },
        "provider": "tenstorrent-direct",
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        # The OpenRouter SDK streams the request body (chunked, no Content-Length).
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            data = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(data)
                data += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_POST(self):
        req = json.loads(self._body() or b"{}")
        if not self.path.rstrip("/").endswith("/responses"):
            self._send(404, json.dumps({"error": {"message": f"unsupported path {self.path}"}}).encode())
            return
        chat = to_chat(req)
        record = {"t": time.time(), "responses_request": req, "chat_request": chat}
        try:
            result = call_upstream(chat)
            record["result"] = result
            response = to_response(req, result)
            event_type = "response.completed" if response["status"] == "completed" else "response.incomplete"
            event = {"type": event_type, "sequence_number": 0, "response": response}
            body = f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()
            self._send(200, body, "text/event-stream")
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:2000]
            record["upstream_error"] = {"status": err.code, "body": detail}
            # A 4xx from the stack is a scored failure, not a transport retry: report it as a
            # failed response so the harness records it instead of retrying forever.
            event = {"type": "response.failed", "sequence_number": 0,
                     "response": {"id": "resp_err", "status": "failed",
                                  "error": {"code": str(err.code), "message": detail}}}
            body = f"event: response.failed\ndata: {json.dumps(event)}\n\n".encode()
            self._send(200, body, "text/event-stream")
        except Exception as exc:  # noqa: BLE001
            record["adapter_error"] = repr(exc)
            self._send(502, json.dumps({"error": {"message": repr(exc)}}).encode())
        finally:
            if LOG:
                with _log_lock, open(LOG, "a") as fh:
                    fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    print(f"adapter on :{PORT} -> {UPSTREAM} model={MODEL} stream={STREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
