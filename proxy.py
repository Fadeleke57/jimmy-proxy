#!/usr/bin/env python3
"""
Proxy server that translates OpenAI-compatible API requests
to chatjimmy.ai's custom format and back.

Usage:
    python proxy.py [--port 4100]

Then point OpenCode at http://localhost:4100/v1
Logs are written to proxy.log (full request/response details).
"""

import json
import time
import uuid
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import ssl
import re

UPSTREAM_URL = "https://chatjimmy.ai/api/chat"
DEFAULT_MODEL = "llama3.1-8B"

MODELS = {
    "llama3.1-8B": "llama3.1-8B",
}


# ---------------------------------------------------------------------------
# Tool-call helpers
# ---------------------------------------------------------------------------

def _first_sentence(text):
    """Return the first sentence (or first 120 chars) of a description."""
    if not text:
        return ""
    # Cut at first period followed by space/newline, or first newline
    for end in (". ", ".\n", "\n"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1].strip()
    return text[:120].strip()


def format_tools_for_prompt(tools, tool_choice=None):
    """Convert OpenAI tool definitions into a Llama-friendly system-prompt section."""
    if not tools:
        return ""

    lines = [
        "",
        "# Tools",
        "When you need a tool, respond with one or more <tool_call> blocks and nothing else.",
        "Format:",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"required_param": "value"}}',
        "</tool_call>",
        "The `arguments` object MUST include all required parameters and only valid JSON.",
        "Do not invent tool results. Tool results will be provided in <tool_result> tags.",
        "",
    ]

    if tool_choice == "none":
        lines.append("Do NOT use tools for this request.")
        lines.append("")
    elif tool_choice == "required":
        lines.append("You MUST call at least one tool.")
        lines.append("")
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fname = tool_choice.get("function", {}).get("name", "")
        if fname:
            lines.append(f"You MUST call '{fname}'.")
            lines.append("")

    # Compact, human-readable signatures
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        name = func.get("name", "")
        desc = _first_sentence(func.get("description", ""))
        params = func.get("parameters", {})

        props = params.get("properties", {})
        required = set(params.get("required", []))

        parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            opt = "" if pname in required else "?"
            parts.append(f"{pname}{opt}: {ptype}")
        sig = ", ".join(parts)
        line = f"- {name}({sig})"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    # Compact JSON schema (strip verbose descriptions to stay within upstream limits)
    try:
        compact_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool["function"]
            params = func.get("parameters", {})
            compact_props = {}
            for pname, pinfo in params.get("properties", {}).items():
                compact_props[pname] = {"type": pinfo.get("type", "string")}
                if "enum" in pinfo:
                    compact_props[pname]["enum"] = pinfo["enum"]
                if "items" in pinfo and isinstance(pinfo["items"], dict):
                    compact_props[pname]["items"] = {"type": pinfo["items"].get("type", "object")}
            compact_tools.append({
                "name": func.get("name", ""),
                "parameters": {
                    "type": "object",
                    "properties": compact_props,
                    "required": params.get("required", []),
                },
            })
        lines.append("")
        lines.append("<tools>")
        lines.append(json.dumps(compact_tools))
        lines.append("</tools>")
    except (TypeError, ValueError):
        pass

    lines.append("")
    return "\n".join(lines)


def _tool_schema_index(tools):
    index = {}
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if name:
            index[name] = func.get("parameters", {}) or {}
    return index


def _default_for_type(ptype):
    if ptype == "string":
        return ""
    if ptype == "integer":
        return 0
    if ptype == "number":
        return 0
    if ptype == "boolean":
        return False
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return ""


def _normalize_tool_args(name, raw_args, schema):
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}

    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []

    for key in required:
        if key not in raw_args or raw_args[key] is None:
            pinfo = props.get(key, {})
            raw_args[key] = _default_for_type(pinfo.get("type", "string"))
        else:
            pinfo = props.get(key, {})
            ptype = pinfo.get("type", "string")
            if ptype == "string" and not isinstance(raw_args[key], str):
                raw_args[key] = str(raw_args[key])
            elif ptype == "integer" and not isinstance(raw_args[key], int):
                try:
                    raw_args[key] = int(raw_args[key])
                except (ValueError, TypeError):
                    raw_args[key] = 0
            elif ptype == "number" and not isinstance(raw_args[key], (int, float)):
                try:
                    raw_args[key] = float(raw_args[key])
                except (ValueError, TypeError):
                    raw_args[key] = 0
            elif ptype == "boolean" and not isinstance(raw_args[key], bool):
                raw_args[key] = bool(raw_args[key])
            elif ptype == "array" and not isinstance(raw_args[key], list):
                raw_args[key] = [raw_args[key]]
            elif ptype == "object" and not isinstance(raw_args[key], dict):
                raw_args[key] = {}

    return raw_args


def _extract_call_objects(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            return obj.get("tool_calls")
        return [obj]
    return []


def parse_tool_calls(content, tools=None):
    """
    Parse <tool_call>…</tool_call> blocks from the model's text.

    Returns (text_without_tags, list_of_openai_tool_call_dicts).
    """
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = pattern.findall(content)

    if not matches:
        return content, []

    tool_calls = []
    schema_index = _tool_schema_index(tools)
    for raw in matches:
        try:
            call = json.loads(raw.strip())
            for item in _extract_call_objects(call):
                if not isinstance(item, dict):
                    continue
                name = (
                    item.get("name")
                    or item.get("tool")
                    or item.get("tool_name")
                    or (item.get("function") or {}).get("name")
                )
                if not name:
                    continue
                arguments = (
                    item.get("arguments")
                    or item.get("parameters")
                    or item.get("args")
                    or item.get("tool_input")
                    or item.get("input")
                )
                if "function" in item and isinstance(item["function"], dict):
                    if arguments is None:
                        arguments = item["function"].get("arguments")
                schema = schema_index.get(name, {})
                arguments = _normalize_tool_args(name, arguments, schema)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                })
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue

    text = pattern.sub("", content).strip()
    return text, tool_calls


def extract_text_content(content):
    """Extract plain text from a message content field (string or list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

# Console logger — minimal
console = logging.getLogger("proxy.console")

# File logger — everything
filelog = logging.getLogger("proxy.file")


def setup_logging(log_file="proxy.log"):
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    console.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    console.addHandler(ch)

    filelog.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    filelog.addHandler(fh)


def log(msg):
    """Log to both console and file."""
    console.info(msg)
    filelog.info(msg)


def logfile(msg):
    """Log to file only."""
    filelog.debug(msg)


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            log(f"GET /v1/models -> {len(MODELS)} model(s)")
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "chatjimmy",
                    }
                    for model_id in MODELS
                ],
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        try:
            openai_req = json.loads(raw)
        except json.JSONDecodeError:
            log("Bad request: invalid JSON")
            self._send_json(400, {"error": "invalid JSON"})
            return

        messages = openai_req.get("messages", [])
        model = openai_req.get("model", DEFAULT_MODEL)
        stream = openai_req.get("stream", False)
        tools = openai_req.get("tools", [])
        tool_choice = openai_req.get("tool_choice", "auto")

        last_content = extract_text_content(
            messages[-1].get("content", "") if messages else ""
        )
        last_preview = last_content[:100] + "..." if len(last_content) > 100 else last_content

        # Console: short summary
        log(f"-> model={model} msgs={len(messages)} stream={stream} tools={len(tools)} | \"{last_preview}\"")

        # File: full incoming request
        logfile("--- INCOMING REQUEST ---")
        logfile(f"Headers: {dict(self.headers)}")
        logfile(f"Body:\n{json.dumps(openai_req, indent=2)}")

        # ----- Build system prompt & chat messages -----
        system_prompt = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = extract_text_content(msg.get("content"))

            if role == "system":
                system_prompt += content + "\n"

            elif role == "assistant" and msg.get("tool_calls"):
                # Re-serialize the assistant's previous tool calls so the
                # model sees what it called last time.
                parts = []
                if content:
                    parts.append(content)
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = func.get("arguments", {})
                    parts.append(
                        "<tool_call>\n"
                        + json.dumps({"name": func.get("name", ""), "arguments": args}, indent=2)
                        + "\n</tool_call>"
                    )
                chat_messages.append({"role": "assistant", "content": "\n".join(parts)})

            elif role == "tool":
                # Tool results → presented as a user message so Llama sees them.
                tool_name = msg.get("name", "unknown")
                tid = msg.get("tool_call_id", "")
                tool_result = {
                    "name": tool_name,
                    "tool_call_id": tid,
                    "content": content,
                }
                chat_messages.append({
                    "role": "user",
                    "content": (
                        "<tool_result>\n"
                        + json.dumps(tool_result, indent=2)
                        + "\n</tool_result>"
                    ),
                })

            else:
                chat_messages.append({"role": role, "content": content})

        # Append tool definitions to the system prompt
        full_system_prompt = system_prompt.strip()
        if tools:
            full_system_prompt += format_tools_for_prompt(tools, tool_choice)

        # ChatJimmy returns empty responses when system prompt exceeds ~30K chars
        MAX_SYSTEM_PROMPT = 28000
        if len(full_system_prompt) > MAX_SYSTEM_PROMPT:
            logfile(f"WARNING: system prompt is {len(full_system_prompt)} chars, truncating to {MAX_SYSTEM_PROMPT}")
            full_system_prompt = full_system_prompt[:MAX_SYSTEM_PROMPT]

        jimmy_payload = {
            "messages": chat_messages,
            "chatOptions": {
                "selectedModel": MODELS.get(model, model),
                "systemPrompt": full_system_prompt,
                "topK": 8,
            },
            "attachment": None,
        }

        # File: translated payload
        logfile("--- TRANSLATED PAYLOAD ---")
        logfile(f"{json.dumps(jimmy_payload, indent=2)}")

        # Forward to chatjimmy
        upstream_start = time.time()
        try:
            req = urllib.request.Request(
                UPSTREAM_URL,
                data=json.dumps(jimmy_payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "Origin": "https://chatjimmy.ai",
                    "Referer": "https://chatjimmy.ai/",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/145.0.0.0 Safari/537.36",
                },
            )
            ctx = ssl.create_default_context()
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            raw_response = resp.read().decode("utf-8")
            elapsed = time.time() - upstream_start
        except Exception as e:
            elapsed = time.time() - upstream_start
            log(f"<- FAILED {elapsed:.2f}s | {e}")
            logfile(f"Upstream error: {e}")
            self._send_json(502, {"error": f"upstream error: {str(e)}"})
            return

        # File: raw upstream response
        logfile("--- RAW UPSTREAM RESPONSE ---")
        logfile(raw_response)

        # Strip stats, parse usage
        content = re.sub(r"<\|stats\|>.*?<\|/stats\|>", "", raw_response, flags=re.DOTALL).strip()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stats_match = re.search(r"<\|stats\|>(.*?)<\|/stats\|>", raw_response, re.DOTALL)
        if stats_match:
            try:
                stats = json.loads(stats_match.group(1))
                usage["prompt_tokens"] = stats.get("prefill_tokens", 0)
                usage["completion_tokens"] = stats.get("decode_tokens", 0)
                usage["total_tokens"] = stats.get("total_tokens", 0)
            except json.JSONDecodeError:
                pass

        # ----- Detect tool calls in model output -----
        text_content, tool_calls_parsed = (
            parse_tool_calls(content, tools) if tools else (content, [])
        )

        if tool_calls_parsed:
            finish_reason = "tool_calls"
            message = {
                "role": "assistant",
                "content": text_content if text_content else None,
                "tool_calls": tool_calls_parsed,
            }
            tc_names = [tc["function"]["name"] for tc in tool_calls_parsed]
            reply_preview = f"[tool_calls: {', '.join(tc_names)}]"
        else:
            finish_reason = "stop"
            message = {"role": "assistant", "content": content}
            reply_preview = content[:100] + "..." if len(content) > 100 else content

        log(f"<- {elapsed:.2f}s {usage['total_tokens']}tok | \"{reply_preview}\"")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            now = int(time.time())
            if tool_calls_parsed:
                # Stream tool-call chunks in OpenAI's format
                chunks = [
                    {"id": completion_id, "object": "chat.completion.chunk",
                     "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                ]
                if text_content:
                    chunks.append(
                        {"id": completion_id, "object": "chat.completion.chunk",
                         "created": now, "model": model,
                         "choices": [{"index": 0, "delta": {"content": text_content}, "finish_reason": None}]})
                for i, tc in enumerate(tool_calls_parsed):
                    chunks.append(
                        {"id": completion_id, "object": "chat.completion.chunk",
                         "created": now, "model": model,
                         "choices": [{"index": 0, "delta": {"tool_calls": [{
                             "index": i,
                             "id": tc["id"],
                             "type": "function",
                             "function": {
                                 "name": tc["function"]["name"],
                                 "arguments": tc["function"]["arguments"],
                             },
                         }]}, "finish_reason": None}]})
                chunks.append(
                    {"id": completion_id, "object": "chat.completion.chunk",
                     "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            else:
                chunks = [
                    {"id": completion_id, "object": "chat.completion.chunk",
                     "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"id": completion_id, "object": "chat.completion.chunk",
                     "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                    {"id": completion_id, "object": "chat.completion.chunk",
                     "created": now, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            self._send_sse(chunks)
        else:
            openai_response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage,
            }
            self._send_json(200, openai_response)

        # File: full outgoing response
        logfile("--- OUTGOING RESPONSE ---")
        if stream:
            for c in chunks:
                logfile(json.dumps(c))
        else:
            logfile(json.dumps(openai_response, indent=2))
        logfile("---")


def main():
    parser = argparse.ArgumentParser(description="ChatJimmy -> OpenAI proxy")
    parser.add_argument("--port", type=int, default=4100, help="Port to listen on")
    parser.add_argument("--log-file", type=str, default="proxy.log", help="Log file path")
    args = parser.parse_args()

    setup_logging(args.log_file)
    log(f"Proxy listening on http://localhost:{args.port}/v1 -> {UPSTREAM_URL}")

    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
