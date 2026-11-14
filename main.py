#!/usr/bin/env python3
# free-model-aggregator-01: one server, two client wires, ten free providers.
#
# What this does, end to end:
#   * reads the trial brief at $MAKEFUN_PROMPT_FILE (three JSON documents:
#     the provider directory, the quality table, the workload manifest);
#   * probes every provider's catalog route in that provider's own wire and
#     builds one catalog document (printed by `--list`, posted to /register,
#     the same object both times);
#   * serves POST /v1/chat/completions and POST /v1/messages (JSON and SSE)
#     plus GET /v1/models in the shape of the caller's auth header;
#   * for each request, ranks every free (provider, model) pair that meets
#     the request's requirements by the quality table's rating, spends the
#     best provider's budget only while it keeps answering, and fails over
#     to the next candidate when a provider refuses (402 / 429 / 5xx);
#   * speaks all three upstream wires (OpenAI, Anthropic, Gemini),
#     translating the request and the answer -- streamed or not -- through
#     one canonical Chat Completions shape;
#   * polls GET $MAKEFUN_ENV_URL/work and exits 0 when the env is done.
#
# Everything is a pure function of the brief, the requests and the
# providers' answers: no wall clock (a fixed `created` is used), no
# randomness, so a replayed trial makes the same provider calls.
#
# Python 3.13, standard library only.

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
FIXED_CREATED = 1767225600  # 2026-01-01T00:00:00Z, a constant, never the clock
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
MAX_BODY = 32 * 1024 * 1024

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_rec(v):
    return isinstance(v, dict)


# --------------------------------------------------------------------------
# The brief


def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents(text):
    """Every JSON object in the brief that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_rec(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _s(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def read_brief(path):
    brief = {"providers": [], "ratings": [], "best": {}, "manifest": [],
             "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief holds no provider directory"
        return brief
    for raw in directory["providers"]:
        if not is_rec(raw):
            continue
        pid = _s(raw.get("id"))
        wire = raw.get("wire")
        if not pid or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _s(raw.get(key)):
                base = _s(raw.get(key))
                break
        brief["providers"].append({
            "id": pid, "name": _s(raw.get("name")) or pid, "wire": wire,
            "auth": auth, "base": base, "key_env": _s(raw.get("key_env")),
            "key": _s(raw.get("key")),
        })
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_rec(raw) or not _s(raw.get("model")) \
                    or not _s(raw.get("provider")):
                continue
            brief["ratings"].append({
                "model": raw["model"], "provider": raw["provider"],
                "rating_milli": raw.get("rating_milli")
                if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context_length": _num_or_none(raw.get("context_length")),
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        for raw in ratings_doc.get("best") or []:
            if is_rec(raw) and _s(raw.get("id")):
                brief["best"][raw["id"]] = raw.get("rating_milli") or 0
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_rec(raw) or not _s(raw.get("id")):
                continue
            req = raw.get("requires") if is_rec(raw.get("requires")) else {}
            requires = {"tools": req.get("tools") is True,
                        "vision": req.get("vision") is True,
                        "json": req.get("json") is True}
            if isinstance(req.get("min_context"), (int, float)):
                requires["min_context"] = req["min_context"]
            brief["manifest"].append({
                "id": raw["id"],
                "api": "anthropic" if raw.get("api") == "anthropic"
                else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": requires,
            })
    return brief


def rating_row(brief, provider_id, model_id):
    for row in brief["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# --------------------------------------------------------------------------
# Probing and the catalog


def join_base(env_url, base):
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_open(req, timeout):
    return urllib.request.urlopen(req, timeout=timeout)


def _zero_priced(model):
    pricing = model.get("pricing")
    if not is_rec(pricing):
        return None
    values = []
    for v in pricing.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
        elif isinstance(v, str) and v.strip():
            try:
                values.append(float(v))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _model_in(wire, raw):
    if not is_rec(raw):
        return None
    if wire == "gemini":
        mid = _s(raw.get("name")).replace("models/", "", 1) \
            if _s(raw.get("name")).startswith("models/") \
            else _s(raw.get("name"))
    else:
        mid = _s(raw.get("id"))
    if not mid:
        return None
    free = raw.get("free") if isinstance(raw.get("free"), bool) else None
    if free is None:
        free = _zero_priced(raw)
    context = _num_or_none(raw.get("context_length")) \
        or _num_or_none(raw.get("context_window")) \
        or _num_or_none(raw.get("inputTokenLimit"))
    return {
        "id": mid,
        "display_name": _s(raw.get("display_name"))
        or _s(raw.get("displayName")) or mid,
        "free": free, "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool)
        else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool)
        else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool)
        else None,
    }


def probe_provider(env_url, provider, timeout=5.0):
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with http_open(req, timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as err:  # refused, non-2xx, not JSON: no traffic for it
        return {"provider": provider, "ok": False, "models": [],
                "error": "GET %s failed: %s" % (url, err)}
    if not is_rec(body):
        return {"provider": provider, "ok": False, "models": [],
                "error": "GET %s answered a non-object" % url}
    listing = body.get("models") if provider["wire"] == "gemini" \
        else body.get("data")
    models = []
    if isinstance(listing, list):
        for raw in listing:
            model = _model_in(provider["wire"], raw)
            if model is not None:
                models.append(model)
    return {"provider": provider, "ok": True, "models": models,
            "error": None}


def model_free(model, row):
    """The quality table is the sealed truth; the listing's word is next."""
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities(model, row):
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": model["context_length"], "tools": model["tools"],
            "vision": model["vision"], "json": model["json"]}


def build_catalog(probes, brief):
    """The one catalog document: --list output and register body alike.

    base_url stays the directory's own path, never joined to the env's
    origin: the origin differs between a live trial and its replay, and
    the catalog must be byte-identical across both.
    """
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(brief, probe["provider"]["id"], model["id"])
            caps = capabilities(model, row)
            models.append({"id": model["id"],
                           "free": model_free(model, row),
                           "context_length": caps["context_length"]})
        providers.append({"id": probe["provider"]["id"],
                          "base_url": probe["provider"]["base"],
                          "key_env": probe["provider"]["key_env"],
                          "wire": probe["provider"]["wire"],
                          "models": models})
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire translation. The canonical shape is OpenAI Chat Completions: every
# client request is rewritten to it, every provider answer is rewritten
# from it, so the three upstream wires and the two client wires meet in
# one place.


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if is_rec(block) and block.get("type") == "text" \
                    and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def arguments_of(value):
    if is_rec(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if is_rec(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def stop_reason_of(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def finish_reason_of(stop):
    if stop == "max_tokens":
        return "length"
    if stop == "tool_use":
        return "tool_calls"
    if stop == "refusal":
        return "content_filter"
    return "stop"


# --- Anthropic client request -> canonical OpenAI request -----------------

def _part_of(block):
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_rec(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" \
                and isinstance(source.get("data"), str):
            media = source.get("media_type") \
                if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s"
                                  % (media, source["data"])}}
        if source.get("type") == "url" \
                and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def anthropic_req_to_openai(body, model):
    messages = []
    system = text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_rec(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user",
                             "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            calls = []
            for block in content:
                if not is_rec(block) or block.get("type") != "tool_use":
                    continue
                calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str)
                    else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name")
                        if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(
                            block["input"] if is_rec(block.get("input"))
                            else {}),
                    },
                })
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text else None}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_rec(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id")
                    if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = _part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) \
            and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = []
    for raw in body.get("tools") or []:
        if not is_rec(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({"type": "function", "function": {
            "name": raw["name"],
            "description": raw.get("description")
            if isinstance(raw.get("description"), str) else "",
            "parameters": raw["input_schema"]
            if is_rec(raw.get("input_schema"))
            else {"type": "object", "properties": {}},
        }})
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_rec(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" \
                and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": choice["name"]}}
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# --- canonical OpenAI request -> Anthropic upstream request ---------------

def openai_req_to_anthropic(body, model):
    system_parts = []
    messages = []
    pending_tool_results = None

    def flush_tools():
        nonlocal pending_tool_results
        if pending_tool_results:
            messages.append({"role": "user",
                             "content": pending_tool_results})
            pending_tool_results = None

    for raw in body.get("messages") or []:
        if not is_rec(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            if pending_tool_results is None:
                pending_tool_results = []
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id")
                if isinstance(raw.get("tool_call_id"), str) else "",
                "content": tool_result_text(content),
            })
            continue
        flush_tools()
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_rec(call) or not is_rec(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id")
                    if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name")
                    if isinstance(fn.get("name"), str) else "",
                    "input": arguments_of(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user (or anything else): text and image parts become blocks
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_rec(part):
                    continue
                if part.get("type") == "text" \
                        and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" \
                        and is_rec(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    if url.startswith("data:") and ";base64," in url:
                        media = url[5:url.index(";")]
                        data = url.split(";base64,", 1)[1]
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": media,
                            "data": data}})
                    else:
                        blocks.append({"type": "image", "source": {
                            "type": "url", "url": url}})
        if blocks:
            messages.append({"role": "user", "content": blocks})
    flush_tools()
    out = {"model": model,
           "max_tokens": body.get("max_tokens")
           if isinstance(body.get("max_tokens"), (int, float)) else 4096,
           "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop"), (str, list)) and body["stop"]:
        out["stop_sequences"] = body["stop"] \
            if isinstance(body["stop"], list) else [body["stop"]]
    tools = []
    for raw in body.get("tools") or []:
        if not is_rec(raw) or not is_rec(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        tools.append({
            "name": fn["name"],
            "description": fn.get("description")
            if isinstance(fn.get("description"), str) else "",
            "input_schema": fn["parameters"] if is_rec(fn.get("parameters"))
            else {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_rec(choice) and is_rec(choice.get("function")) \
            and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool",
                              "name": choice["function"]["name"]}
    if body.get("stream") is True:
        out["stream"] = True
    return out


# --- canonical OpenAI request -> Gemini upstream request ------------------

def openai_req_to_gemini(body, model):
    system_parts = []
    contents = []
    # Gemini matches a functionResponse to its call by name, and the
    # canonical tool message carries only the call id: recover each
    # id's name from the assistant turns that made the calls.
    call_names = {}
    for raw in body.get("messages") or []:
        if not is_rec(raw):
            continue
        for call in raw.get("tool_calls") or []:
            if is_rec(call) and is_rec(call.get("function")) \
                    and isinstance(call.get("id"), str):
                call_names[call["id"]] = call["function"].get("name") or ""
    for raw in body.get("messages") or []:
        if not is_rec(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            text = tool_result_text(content)
            try:
                parsed = json.loads(text) if text else None
            except ValueError:
                parsed = None
            response = parsed if is_rec(parsed) else {"result": text}
            name = call_names.get(raw.get("tool_call_id"), "")
            contents.append({"role": "user", "parts": [{
                "functionResponse": {"name": name, "response": response}}]})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if not is_rec(part):
                    continue
                if part.get("type") == "text" \
                        and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url" \
                        and is_rec(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    if url.startswith("data:") and ";base64," in url:
                        media = url[5:url.index(";")]
                        data = url.split(";base64,", 1)[1]
                        parts.append({"inlineData": {"mimeType": media,
                                                     "data": data}})
                    else:
                        parts.append({"fileData": {"fileUri": url}})
        if role == "assistant":
            for call in raw.get("tool_calls") or []:
                if not is_rec(call) or not is_rec(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name")
                    if isinstance(fn.get("name"), str) else "",
                    "args": arguments_of(fn.get("arguments"))}})
            contents.append({"role": "model", "parts": parts})
        elif parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    declarations = []
    for raw in body.get("tools") or []:
        if not is_rec(raw) or not is_rec(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        declarations.append({
            "name": fn["name"],
            "description": fn.get("description")
            if isinstance(fn.get("description"), str) else "",
            "parameters": fn["parameters"] if is_rec(fn.get("parameters"))
            else {"type": "object", "properties": {}},
        })
    if declarations:
        out["tools"] = [{"functionDeclarations": declarations}]
    gen = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        gen["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        gen["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        gen["topP"] = body["top_p"]
    if isinstance(body.get("stop"), (str, list)) and body["stop"]:
        gen["stopSequences"] = body["stop"] \
            if isinstance(body["stop"], list) else [body["stop"]]
    fmt = body.get("response_format")
    if is_rec(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        gen["responseMimeType"] = "application/json"
    if gen:
        out["generationConfig"] = gen
    return out


# --- provider answers -> canonical OpenAI answer --------------------------

def _usage(prompt, completion):
    prompt = prompt if isinstance(prompt, (int, float)) else 0
    completion = completion if isinstance(completion, (int, float)) else 0
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion),
            "total_tokens": int(prompt) + int(completion)}


def anthropic_resp_to_openai(body, model):
    content = []
    calls = []
    for block in body.get("content") or []:
        if not is_rec(block):
            continue
        if block.get("type") == "text" \
                and isinstance(block.get("text"), str):
            content.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str)
                else "toolu_unknown",
                "type": "function",
                "function": {
                    "name": block.get("name")
                    if isinstance(block.get("name"), str) else "",
                    "arguments": json.dumps(block.get("input")
                                            if is_rec(block.get("input"))
                                            else {}),
                },
            })
    message = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = calls
    usage = body.get("usage") if is_rec(body.get("usage")) else {}
    return {
        "id": body.get("id") if isinstance(body.get("id"), str)
        else "msg_aggregate",
        "object": "chat.completion", "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason_of(
                         body.get("stop_reason"))}],
        "usage": _usage(usage.get("input_tokens"),
                        usage.get("output_tokens")),
    }


def gemini_resp_to_openai(body, model):
    candidates = body.get("candidates") if isinstance(
        body.get("candidates"), list) else []
    first = candidates[0] if candidates and is_rec(candidates[0]) else {}
    content_obj = first.get("content") if is_rec(first.get("content")) else {}
    text_parts = []
    calls = []
    for part in content_obj.get("parts") or []:
        if not is_rec(part):
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif is_rec(part.get("functionCall")):
            fc = part["functionCall"]
            calls.append({
                "id": "call_%d" % len(calls),
                "type": "function",
                "function": {
                    "name": fc.get("name")
                    if isinstance(fc.get("name"), str) else "",
                    "arguments": json.dumps(fc.get("args")
                                            if is_rec(fc.get("args"))
                                            else {}),
                },
            })
    message = {"role": "assistant",
               "content": "".join(text_parts) or None}
    if calls:
        message["tool_calls"] = calls
    finish = first.get("finishReason")
    finish_reason = {"STOP": "stop", "MAX_TOKENS": "length",
                     "SAFETY": "content_filter"}.get(finish, "stop")
    usage = body.get("usageMetadata") if is_rec(
        body.get("usageMetadata")) else {}
    return {
        "id": "chatcmpl_aggregate", "object": "chat.completion",
        "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason}],
        "usage": _usage(usage.get("promptTokenCount"),
                        usage.get("candidatesTokenCount")),
    }


# --- canonical OpenAI answer -> Anthropic client answer -------------------

def openai_resp_to_anthropic(body, model):
    choices = body.get("choices") if isinstance(body.get("choices"), list) \
        else []
    choice = choices[0] if choices and is_rec(choices[0]) else {}
    message = choice.get("message") if is_rec(choice.get("message")) else {}
    content = []
    text = message.get("content") if isinstance(
        message.get("content"), str) else text_of(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not is_rec(call) or not is_rec(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call.get("id") if isinstance(call.get("id"), str)
            else "toolu_unknown",
            "name": fn.get("name") if isinstance(fn.get("name"), str)
            else "",
            "input": arguments_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_rec(body.get("usage")) else {}
    return {
        "id": "msg_" + (body.get("id") if isinstance(body.get("id"), str)
                        else "aggregate"),
        "type": "message", "role": "assistant", "model": model,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                  "output_tokens": int(usage.get("completion_tokens") or 0)},
    }


# --------------------------------------------------------------------------
# Provider calls. One canonical chunk shape flows back from every wire:
# an OpenAI chat.completion.chunk dict.


def _provider_headers(provider, stream=False):
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if stream:
        headers["Accept"] = "text/event-stream"
    return headers


def _provider_url(provider, model, stream):
    base = join_base(ENV_URL, provider["base"])
    if provider["wire"] == "openai":
        return base + "/v1/chat/completions"
    if provider["wire"] == "anthropic":
        return base + "/v1/messages"
    verb = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    return "%s/v1beta/models/%s%s" % (base, model, verb)


def _provider_payload(provider, canonical, model):
    if provider["wire"] == "openai":
        body = dict(canonical)
        body["model"] = model
        if body.get("stream") is True:
            # A streamed answer must still carry its final usage: ask the
            # provider for it even when the client did not think to.
            options = dict(body.get("stream_options") or {})
            options["include_usage"] = True
            body["stream_options"] = options
        return body
    if provider["wire"] == "anthropic":
        return openai_req_to_anthropic(canonical, model)
    return openai_req_to_gemini(canonical, model)


class ProviderFailure(Exception):
    def __init__(self, status, text=""):
        super().__init__("provider answered HTTP %s" % status)
        self.status = status
        self.text = text


def call_provider_json(provider, canonical, model):
    """One non-streamed provider call, answered as a canonical body."""
    payload = _provider_payload(provider, canonical, model)
    req = urllib.request.Request(
        _provider_url(provider, model, False),
        data=json.dumps(payload).encode("utf-8"),
        headers=_provider_headers(provider), method="POST")
    try:
        with http_open(req, 30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise ProviderFailure(err.code, err.read().decode("utf-8", "replace"))
    if provider["wire"] == "openai":
        body["model"] = model
        body["created"] = FIXED_CREATED
        return body
    if provider["wire"] == "anthropic":
        return anthropic_resp_to_openai(body, model)
    return gemini_resp_to_openai(body, model)


def _sse_data_lines(resp):
    """The data payloads of a server-sent event stream, in order."""
    event_data = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if event_data:
                yield "\n".join(event_data)
                event_data = []
            continue
        if line.startswith("data:"):
            event_data.append(line[5:].lstrip())
    if event_data:
        yield "\n".join(event_data)


def _chunk(model, delta, finish=None, usage=None, cid="chatcmpl_aggregate"):
    chunk = {"id": cid, "object": "chat.completion.chunk",
             "created": FIXED_CREATED, "model": model,
             "choices": [{"index": 0, "delta": delta,
                          "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _stream_openai(resp, model):
    for payload in _sse_data_lines(resp):
        if payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if is_rec(chunk):
            chunk["model"] = model
            chunk["created"] = FIXED_CREATED
            yield chunk


def _stream_anthropic(resp, model):
    blocks = {}   # anthropic block index -> ("text", None) | ("tool", slot)
    slot = 0
    usage_in = 0
    usage_out = 0
    event = None
    buffer = []

    def frames():
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("event:"):
                event_holder[0] = line[6:].strip()
            elif line.startswith("data:"):
                buffer.append(line[5:].lstrip())
            elif line == "" and buffer:
                payload = "\n".join(buffer)
                buffer.clear()
                yield event_holder[0], payload

    event_holder = [None]
    for ev, payload in frames():
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not is_rec(data):
            continue
        if ev == "message_start":
            msg = data.get("message") if is_rec(data.get("message")) else {}
            usage = msg.get("usage") if is_rec(msg.get("usage")) else {}
            usage_in = usage.get("input_tokens") or 0
            yield _chunk(model, {"role": "assistant", "content": ""},
                         cid=msg.get("id") or "msg_aggregate")
        elif ev == "content_block_start":
            block = data.get("content_block") or {}
            idx = data.get("index") or 0
            if block.get("type") == "tool_use":
                blocks[idx] = ("tool", slot)
                slot += 1
                yield _chunk(model, {"tool_calls": [{
                    "index": blocks[idx][1],
                    "id": block.get("id") or "toolu_unknown",
                    "type": "function",
                    "function": {"name": block.get("name") or "",
                                 "arguments": ""}}]})
            else:
                blocks[idx] = ("text", None)
        elif ev == "content_block_delta":
            idx = data.get("index") or 0
            delta = data.get("delta") or {}
            kind = blocks.get(idx, ("text", None))
            if delta.get("type") == "text_delta" \
                    and isinstance(delta.get("text"), str):
                yield _chunk(model, {"content": delta["text"]})
            elif delta.get("type") == "input_json_delta" \
                    and kind[0] == "tool":
                yield _chunk(model, {"tool_calls": [{
                    "index": kind[1],
                    "function": {"arguments":
                                 delta.get("partial_json") or ""}}]})
        elif ev == "message_delta":
            usage = data.get("usage") if is_rec(data.get("usage")) else {}
            usage_out = usage.get("output_tokens") or usage_out
            delta = data.get("delta") or {}
            yield _chunk(model, {}, finish=finish_reason_of(
                delta.get("stop_reason")),
                usage=_usage(usage_in, usage_out))
        elif ev == "message_stop":
            return


def _stream_gemini(resp, model):
    call_count = [0]
    for payload in _sse_data_lines(resp):
        try:
            body = json.loads(payload)
        except ValueError:
            continue
        if not is_rec(body):
            continue
        candidates = body.get("candidates") or []
        first = candidates[0] if candidates and is_rec(candidates[0]) \
            else {}
        content = first.get("content") if is_rec(first.get("content")) \
            else {}
        delta = {}
        for part in content.get("parts") or []:
            if not is_rec(part):
                continue
            if isinstance(part.get("text"), str) and part["text"]:
                delta["content"] = delta.get("content", "") + part["text"]
            elif is_rec(part.get("functionCall")):
                fc = part["functionCall"]
                delta.setdefault("tool_calls", []).append({
                    "index": call_count[0], "id": "call_%d" % call_count[0],
                    "type": "function",
                    "function": {"name": fc.get("name") or "",
                                 "arguments": json.dumps(
                                     fc.get("args")
                                     if is_rec(fc.get("args")) else {})}})
                call_count[0] += 1
        finish = first.get("finishReason")
        finish_reason = {"STOP": "stop", "MAX_TOKENS": "length",
                         "SAFETY": "content_filter"}.get(finish)
        usage = None
        meta = body.get("usageMetadata")
        if is_rec(meta) and meta.get("candidatesTokenCount") is not None:
            usage = _usage(meta.get("promptTokenCount"),
                           meta.get("candidatesTokenCount"))
        if delta or finish_reason or usage:
            yield _chunk(model, delta, finish=finish_reason, usage=usage)


def open_provider_stream(provider, canonical, model):
    """Open a streamed provider call; returns (response, chunk iterator).

    Raises ProviderFailure before yielding anything when the provider
    refuses, so the caller can fail over cleanly.
    """
    payload = _provider_payload(provider, canonical, model)
    req = urllib.request.Request(
        _provider_url(provider, model, True),
        data=json.dumps(payload).encode("utf-8"),
        headers=_provider_headers(provider, stream=True), method="POST")
    try:
        resp = http_open(req, 30)
    except urllib.error.HTTPError as err:
        raise ProviderFailure(err.code, err.read().decode("utf-8", "replace"))
    if provider["wire"] == "openai":
        return resp, _stream_openai(resp, model)
    if provider["wire"] == "anthropic":
        return resp, _stream_anthropic(resp, model)
    return resp, _stream_gemini(resp, model)


# --------------------------------------------------------------------------
# Routing: rank the free candidates, spend budgets wisely, fail over.


def meets(requires, caps):
    if requires.get("tools") and caps["tools"] is not True:
        return False
    if requires.get("vision") and caps["vision"] is not True:
        return False
    if requires.get("json") and caps["json"] is not True:
        return False
    if requires.get("min_context") is not None:
        if caps["context_length"] is None \
                or caps["context_length"] < requires["min_context"]:
            return False
    return True


class Router:
    def __init__(self, brief, probes):
        self.brief = brief
        self.options = []  # (provider order, probe, model, caps, rating)
        for order, probe in enumerate(probes):
            if not probe["ok"]:
                continue
            for model in probe["models"]:
                row = rating_row(brief, probe["provider"]["id"],
                                 model["id"])
                if not model_free(model, row):
                    continue
                self.options.append((order, probe, model,
                                     capabilities(model, row),
                                     row["rating_milli"] if row else 0))
        self.state = {}  # provider id -> {"calls": n, "fails": n, "dead": b}

    def _provider_state(self, pid):
        return self.state.setdefault(pid, {"calls": 0, "fails": 0,
                                           "dead": False})

    def candidates(self, requires):
        ranked = []
        for order, probe, model, caps, rating in self.options:
            if not meets(requires, caps):
                continue
            state = self._provider_state(probe["provider"]["id"])
            effective = rating
            if state["dead"]:
                effective -= 200
            elif state["fails"] >= 2:
                effective -= 40
            ranked.append((-effective, order, model["id"], probe, model))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(probe, model) for _e, _o, _m, probe, model in ranked]

    def note_call(self, provider_id):
        self._provider_state(provider_id)["calls"] += 1

    def note_failure(self, provider_id, status):
        state = self._provider_state(provider_id)
        state["fails"] += 1
        if status == 402 or state["fails"] >= 3:
            state["dead"] = True

    def note_success(self, provider_id):
        self._provider_state(provider_id)["fails"] = 0


# --------------------------------------------------------------------------
# The HTTP server


class Entry:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.router = Router(brief, probes)
        self.manifest_index = 0
        self.lock = threading.Lock()

    def served_models(self):
        out = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in sorted(probe["models"], key=lambda m: m["id"]):
                row = rating_row(self.brief, probe["provider"]["id"],
                                 model["id"])
                if model_free(model, row):
                    out.append({"id": model["id"],
                                "provider": probe["provider"]["id"],
                                "display_name": model["display_name"]})
        return out

    def requirements_for(self, body, client_api):
        manifest = self.brief["manifest"]
        entry = manifest[self.manifest_index] \
            if self.manifest_index < len(manifest) else None
        self.manifest_index += 1
        requires = dict(entry["requires"]) if entry else {}
        # Union with what the body itself demands: a body carrying tools,
        # images or a JSON response format needs a model that can do it,
        # whatever the manifest row said.
        messages = body.get("messages") or []
        vision = False
        for msg in messages:
            if not is_rec(msg):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not is_rec(block):
                        continue
                    if block.get("type") in ("image", "image_url"):
                        vision = True
                    if is_rec(block.get("image_url")):
                        vision = True
        if body.get("tools"):
            requires["tools"] = True
        if vision:
            requires["vision"] = True
        fmt = body.get("response_format")
        if is_rec(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
        return requires


ENTRY = None


def _send(handler, status, body_bytes, content_type):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    handler.wfile.write(body_bytes)


def _send_json(handler, status, obj):
    _send(handler, status, json.dumps(obj).encode("utf-8"),
          "application/json")


def _openai_error(message, etype, code=None):
    return {"error": {"message": message, "type": etype, "code": code}}


def _anthropic_error(etype, message):
    return {"type": "error", "error": {"type": etype, "message": message}}


def _send_error(handler, api, status, etype, message):
    if api == "anthropic":
        _send_json(handler, status, _anthropic_error(etype, message))
    else:
        _send_json(handler, status, _openai_error(message, etype))


def _models_response(handler):
    models = ENTRY.served_models()
    if handler.headers.get("x-api-key") is not None:
        _send_json(handler, 200, {
            "data": [{"type": "model", "id": m["id"],
                      "display_name": m["display_name"],
                      "created_at": FIXED_CREATED_AT} for m in models],
            "has_more": False})
    else:
        _send_json(handler, 200, {
            "object": "list",
            "data": [{"id": m["id"], "object": "model",
                      "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]})


def _serve_stream(handler, api, canonical, candidates, requested_model):
    """Try candidates in order; once bytes flow, the die is cast."""
    last_failure = None
    for probe, model in candidates[:6]:
        provider = probe["provider"]
        ENTRY.router.note_call(provider["id"])
        try:
            resp, chunks = open_provider_stream(provider, canonical,
                                                model["id"])
        except ProviderFailure as err:
            ENTRY.router.note_failure(provider["id"], err.status)
            last_failure = err
            continue
        except Exception as err:  # connection failed before any answer
            ENTRY.router.note_failure(provider["id"], 503)
            last_failure = err
            continue
        ENTRY.router.note_success(provider["id"])
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        if api == "openai":
            _write_openai_stream(handler, chunks)
        else:
            _write_anthropic_stream(handler, chunks, requested_model)
        try:
            resp.close()
        except Exception:
            pass
        return True
    if last_failure is not None:
        log("all candidates refused; last: %s" % last_failure)
    return False


def _write_openai_stream(handler, chunks):
    sent_usage = False
    last_usage = None
    for chunk in chunks:
        if is_rec(chunk.get("usage")):
            sent_usage = True
            last_usage = chunk["usage"]
        handler.wfile.write(("data: " + json.dumps(chunk) + "\n\n")
                            .encode("utf-8"))
        handler.wfile.flush()
    if not sent_usage and last_usage is not None:
        tail = {"id": "chatcmpl_aggregate", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": "",
                "choices": [], "usage": last_usage}
        handler.wfile.write(("data: " + json.dumps(tail) + "\n\n")
                            .encode("utf-8"))
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def _sse(event, data):
    head = ("event: %s\n" % event) if event else ""
    return (head + "data: " + json.dumps(data) + "\n\n").encode("utf-8")


def _write_anthropic_stream(handler, chunks, model):
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [_sse("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message",
                        "role": "assistant", "model": model, "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens,
                                  "output_tokens": 0}}})]

    def close_events():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(_sse("content_block_stop",
                            {"type": "content_block_stop",
                             "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(_sse("content_block_stop",
                            {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop):
        nonlocal finished
        if finished:
            return []
        finished = True
        out = close_events()
        out.append(_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}}))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out

    for chunk in chunks:
        if is_rec(chunk.get("usage")):
            input_tokens = chunk["usage"].get("prompt_tokens") or input_tokens
            output_tokens = chunk["usage"].get("completion_tokens") \
                or output_tokens
        choices = chunk.get("choices") or []
        choice = choices[0] if choices and is_rec(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if is_rec(choice.get("delta")) else {}
        for event in start_events():
            handler.wfile.write(event)
        if isinstance(delta.get("content"), str) and delta["content"]:
            if text_index == -1:
                text_index = next_index
                next_index += 1
                handler.wfile.write(_sse("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            handler.wfile.write(_sse("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]}}))
        for call in delta.get("tool_calls") or []:
            if not is_rec(call):
                continue
            upstream_index = call.get("index") \
                if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    handler.wfile.write(_sse("content_block_stop", {
                        "type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_rec(call.get("function")) \
                    else {}
                handler.wfile.write(_sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.get("id")
                        if isinstance(call.get("id"), str)
                        and call.get("id") else "toolu_%d" % index,
                        "name": fn.get("name") or "", "input": {}}}))
            fn = call.get("function") if is_rec(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                handler.wfile.write(_sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn["arguments"]}}))
        handler.wfile.flush()
        if choice.get("finish_reason"):
            pending_stop = stop_reason_of(choice["finish_reason"])
    for event in start_events():
        handler.wfile.write(event)
    for event in finish_events(pending_stop or "end_turn"):
        handler.wfile.write(event)
    handler.wfile.flush()


def _serve_json(handler, api, canonical, candidates, requested_model):
    last_failure = None
    for probe, model in candidates[:6]:
        provider = probe["provider"]
        ENTRY.router.note_call(provider["id"])
        try:
            answer = call_provider_json(provider, canonical, model["id"])
        except ProviderFailure as err:
            ENTRY.router.note_failure(provider["id"], err.status)
            last_failure = err
            continue
        except Exception as err:
            ENTRY.router.note_failure(provider["id"], 503)
            last_failure = err
            continue
        ENTRY.router.note_success(provider["id"])
        if api == "openai":
            _send_json(handler, 200, answer)
        else:
            _send_json(handler, 200,
                       openai_resp_to_anthropic(answer, requested_model))
        return True
    if last_failure is not None:
        log("all candidates refused; last: %s" % last_failure)
    return False


def _handle_api(handler, api):
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(min(length, MAX_BODY)) if length else b""
    try:
        body = json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        body = None
    if not is_rec(body):
        _send_error(handler, api, 400, "invalid_request_error",
                    "the request body is not one JSON object")
        return
    with ENTRY.lock:
        requires = ENTRY.requirements_for(body, api)
        candidates = ENTRY.router.candidates(requires)
        if not candidates:
            _send_error(handler, api, 503,
                        "overloaded_error" if api == "anthropic"
                        else "no_free_model",
                        "no free model meets this request")
            return
        if api == "openai":
            canonical = dict(body)
            stream = body.get("stream") is True
            requested = candidates[0][1]["id"]
        else:
            canonical = anthropic_req_to_openai(body, candidates[0][1]["id"])
            stream = body.get("stream") is True
            requested = body.get("model") \
                if isinstance(body.get("model"), str) and body.get("model") \
                else candidates[0][1]["id"]
        if stream:
            ok = _serve_stream(handler, api, canonical, candidates,
                               requested)
        else:
            ok = _serve_json(handler, api, canonical, candidates, requested)
    if not ok:
        _send_error(handler, api, 503,
                    "overloaded_error" if api == "anthropic"
                    else "upstream_error",
                    "every free provider refused this request")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            _models_response(self)
        else:
            _send_error(self, "openai", 404, "not_found",
                        "no route GET %s" % self.path)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            _handle_api(self, "openai")
        elif path == "/v1/messages":
            _handle_api(self, "anthropic")
        else:
            _send_error(self, "anthropic" if path == "/v1/messages"
                        else "openai", 404, "not_found",
                        "no route POST %s" % self.path)


# --------------------------------------------------------------------------
# Startup, registration, and the done poll


def _poll_done():
    import time
    while True:
        try:
            req = urllib.request.Request(ENV_URL + "/work", method="GET")
            with http_open(req, 2) as resp:
                work = json.loads(resp.read().decode("utf-8"))
            if is_rec(work) and work.get("done") is True:
                os._exit(0)
        except Exception:
            pass
        time.sleep(0.1)


def main():
    global ENTRY
    brief = read_brief(PROMPT_FILE)
    if brief["error"]:
        log(brief["error"])
    probes = []
    if ENV_URL:
        threads = []
        # Probe concurrently, but rebuild the probe list in the brief's
        # own order afterwards: completion order must not leak into the
        # catalog, or a replay could register a different document.
        by_id = {}
        lock = threading.Lock()

        def run_probe_ordered(provider):
            result = probe_provider(ENV_URL, provider)
            with lock:
                by_id[provider["id"]] = result

        for provider in brief["providers"]:
            thread = threading.Thread(target=run_probe_ordered,
                                      args=(provider,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        probes = [by_id[p["id"]] for p in brief["providers"]
                  if p["id"] in by_id]
        for probe in probes:
            if not probe["ok"] and probe["error"]:
                log(probe["error"])
    catalog = build_catalog(probes, brief)
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    ENTRY = Entry(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns: restart")
        return 1
    register = urllib.request.Request(
        ENV_URL + "/register",
        data=json.dumps({"protocol_version": PROTOCOL_VERSION,
                         "port": port, "catalog": catalog}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with http_open(register, 5) as resp:
        reply = json.loads(resp.read().decode("utf-8"))
    log("registered on port %d: %s" % (port, reply))
    threading.Thread(target=_poll_done, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
