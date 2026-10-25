"""Offline tests for the direct Anthropic backend: translation, caching, thinking, streaming."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx2

from ctxagent import accounting, anthropic_api as aa, cache_policy, llm


class NS(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: (v.model_dump() if isinstance(v, NS) else v) for k, v in vars(self).items()
                if not (exclude_none and v is None)}


def ev(type, **kw):
    return NS(type=type, **kw)


class FakeStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def bad_request(message):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(400, request=request)
    return anthropic.BadRequestError(message, response=response, body=None)


THINKING = [{"type": "reasoning.text", "text": "", "signature": "sig-1", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "opaque", "format": "anthropic-claude-v1", "index": 1},
            {"type": "reasoning.text", "text": "other", "signature": "x", "format": "openai-responses-v1", "index": 2}]


def history():
    return [
        {"role": "system", "content": "You are ctxagent. " * 50},
        {"role": "user", "content": "Read both files."},
        {"role": "assistant", "content": None, "reasoning_details": THINKING, "tool_calls": [
            {"id": "call:0|a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            {"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "call:0|a", "content": "A"},
        {"role": "tool", "tool_call_id": "toolu_1", "content": ""},
        {"role": "user", "content": [{"type": "text", "text": "Tool-provided visual evidence"},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks, now summarise."},
    ]


class ConvertTests(unittest.TestCase):
    def test_shapes(self):
        system, msgs = aa.convert_messages(history())
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant", "user"])
        assistant = msgs[1]["content"]
        # Only Anthropic-format reasoning is replayed, first, in order.
        self.assertEqual([b["type"] for b in assistant], ["thinking", "redacted_thinking", "tool_use", "tool_use"])
        self.assertEqual(assistant[2]["id"], "call_0_a")
        self.assertEqual(assistant[2]["input"], {"path": "a"})
        self.assertEqual(assistant[3]["input"], {"_unparsed_arguments": "not json"})
        # Parallel results share ONE user message, results first, then the media.
        results = msgs[2]["content"]
        self.assertEqual([b["type"] for b in results], ["tool_result", "tool_result", "text", "image"])
        self.assertEqual(results[0]["tool_use_id"], "call_0_a")
        self.assertEqual(results[1]["content"], [{"type": "text", "text": "(empty)"}])
        self.assertEqual(results[3]["source"], {"type": "base64", "media_type": "image/png", "data": "AAAA"})

    def test_thinking_can_be_stripped(self):
        _, msgs = aa.convert_messages(history(), keep_thinking=False)
        self.assertNotIn("thinking", [b["type"] for b in msgs[1]["content"]])

    def test_tools(self):
        tools = [{"type": "function", "function": {"name": "read_file", "description": "d",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        self.assertEqual(aa.convert_tools(tools), [{"name": "read_file", "description": "d",
                                                    "input_schema": {"type": "object", "properties": {}}}])


class RequestTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(aa, "_REJECTED", set()); p.start(); self.addCleanup(p.stop)

    def build(self, model="claude-opus-5-5", effort="high", frozen=6, ttl="5m"):
        return aa.build_request(model=model, messages=history(), tools=None, max_tokens=1000,
                                reasoning_effort=effort, frozen_msg_count=frozen,
                                previous_cache_anchor=None, cache_ttl=ttl)

    def test_cache_breakpoints(self):
        kw = self.build()
        self.assertEqual(kw["cache_control"], {"type": "ephemeral"})  # moving tail
        self.assertEqual(kw["system"][-1]["cache_control"], {"type": "ephemeral"})
        marked = [b for m in kw["messages"] for b in m["content"] if b.get("cache_control")]
        self.assertEqual(len(marked), 1)  # end of the frozen prefix
        self.assertLessEqual(len(marked) + 2, 4)
        self.assertEqual(self.build(ttl="1h")["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(cache_policy.cache_family("claude-sonnet-5-5"), "anthropic")

    def test_opus_55(self):
        kw = self.build()
        self.assertEqual(kw["thinking"], {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
        self.assertEqual(kw["output_config"], {"effort": "high"})
        self.assertEqual(kw["fallbacks"], "default")
        self.assertEqual(set(kw["betas"]), {aa.BLOCK_BINDING_BETA, aa.FALLBACK_BETA})
        self.assertNotIn("temperature", kw)
        self.assertIn("thinking", [b["type"] for b in kw["messages"][1]["content"]])
        # Opus 5.5 cannot disable thinking: "off" is lowest effort instead.
        off = self.build(effort="off")
        self.assertEqual(off["thinking"]["type"], "adaptive")
        self.assertEqual(off["output_config"], {"effort": "low"})

    def test_off_variants(self):
        sonnet = self.build("claude-sonnet-5-5", "off")
        self.assertEqual(sonnet["thinking"], {"type": "between_tools"})
        self.assertNotIn("fallbacks", sonnet)
        self.assertNotIn("thinking", [b["type"] for b in sonnet["messages"][1]["content"]])
        self.assertEqual(self.build("claude-opus-4-8", "off")["thinking"], {"type": "disabled"})
        old = self.build("claude-haiku-4-5", "high")
        self.assertNotIn("thinking", old)
        self.assertNotIn("output_config", old)

    def test_rejected_features_are_dropped(self):
        aa._REJECTED.update({"fallbacks", "block_binding"})
        kw = self.build()
        self.assertNotIn("fallbacks", kw)
        self.assertNotIn("betas", kw)
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertNotIn("thinking", [b["type"] for b in kw["messages"][1]["content"]])


def stream_events(*blocks, stop="end_turn", usage_out=12):
    events = [ev("message_start", message=NS(id="msg_1", model="claude-opus-5-5", usage=NS(
        input_tokens=10, cache_read_input_tokens=1000, cache_creation_input_tokens=200, output_tokens=1)))]
    for i, (start, deltas) in enumerate(blocks):
        events.append(ev("content_block_start", index=i, content_block=NS(**start)))
        events += [ev("content_block_delta", index=i, delta=NS(**d)) for d in deltas]
        events.append(ev("content_block_stop", index=i))
    events.append(ev("message_delta", delta=NS(stop_reason=stop, stop_details=None), usage=NS(output_tokens=usage_out)))
    events.append(ev("message_stop"))
    return events


async def collect(gen):
    return [e async for e in gen]


class StreamTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(aa, "_REJECTED", set()); p.start(); self.addCleanup(p.stop)

    def run_stream(self, client, **kw):
        with patch.object(aa, "_get_client", return_value=client):
            return asyncio.run(collect(aa.stream_chat(model="claude-opus-5-5", messages=history(), **kw)))

    def test_text_thinking_tools_usage(self):
        stream = FakeStream(stream_events(
            ({"type": "thinking", "thinking": ""}, [{"type": "thinking_delta", "thinking": ""},
                                                    {"type": "signature_delta", "signature": "SIG"}]),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "Hel"}, {"type": "text_delta", "text": "lo"}]),
            ({"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {}},
             [{"type": "input_json_delta", "partial_json": '{"path":'}, {"type": "input_json_delta", "partial_json": ' "x"}'}]),
            stop="tool_use"))
        events = self.run_stream(FakeClient(stream))
        self.assertTrue(stream.closed)
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "text"), "Hello")
        usage = next(e for e in events if e["type"] == "usage")
        self.assertEqual(usage["usage"]["prompt_tokens"], 1210)
        self.assertEqual(usage["usage"]["prompt_tokens_details"], {"cached_tokens": 1000, "cache_write_tokens": 200})
        self.assertEqual(usage["usage"]["completion_tokens"], 12)
        reasoning = next(e for e in events if e["type"] == "reasoning_details")["reasoning_details"]
        self.assertEqual(reasoning[0]["signature"], "SIG")
        self.assertEqual(reasoning[0]["format"], "anthropic-claude-v1")
        calls = next(e for e in events if e["type"] == "tool_calls")["tool_calls"]
        self.assertEqual(calls[0]["function"], {"name": "read_file", "arguments": '{"path": "x"}'})
        self.assertEqual(events[-1], {"type": "done", "finish_reason": "tool_calls"})
        # Round trip: the stored reasoning replays as a native thinking block.
        self.assertEqual(aa._thinking_blocks(reasoning)[0], {"type": "thinking", "thinking": "", "signature": "SIG"})

    def test_fallback_boundary_discards_pre_boundary_thinking_and_tools(self):
        stream = FakeStream(stream_events(
            ({"type": "thinking", "thinking": ""}, [{"type": "signature_delta", "signature": "OLD"}]),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "partial "}]),
            ({"type": "fallback"}, []),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "rest"}])))
        events = self.run_stream(FakeClient(stream))
        self.assertFalse([e for e in events if e["type"] == "reasoning_details"])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "text"), "partial rest")

    def test_refusal_is_an_error(self):
        events = self.run_stream(FakeClient(FakeStream(stream_events(stop="refusal"))))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("refusal", events[-1]["error"])
        self.assertTrue(any(e["type"] == "usage" for e in events))

    def test_rejected_fallbacks_retry_once_without(self):
        client = FakeClient(bad_request("fallbacks: not available for this organization"),
                            FakeStream(stream_events(({"type": "text", "text": ""}, [{"type": "text_delta", "text": "ok"}]))))
        events = self.run_stream(client)
        self.assertIn("fallbacks", client.calls[0])
        self.assertNotIn("fallbacks", client.calls[1])
        self.assertEqual(events[-1]["finish_reason"], "stop")

    def test_other_400_is_reported(self):
        events = self.run_stream(FakeClient(bad_request("messages.1: bad")))
        self.assertEqual(events, [{"type": "error", "error": "HTTP 400: messages.1: bad"}])

    def test_complete_uses_stream_without_cache_markers(self):
        client = FakeClient(FakeStream(stream_events(({"type": "text", "text": ""}, [{"type": "text_delta", "text": "{}"}]))))
        seen = []

        async def cb(raw, **meta):
            seen.append(meta["provider"])
        with patch.object(aa, "_get_client", return_value=client):
            text = asyncio.run(llm.complete(model="claude-haiku-5-5", messages=history(), usage_callback=cb,
                                            response_format={"type": "json_object"}))
        self.assertEqual(text, "{}")
        self.assertEqual(seen, ["Anthropic"])
        self.assertNotIn("cache_control", client.calls[0])


class ClientTests(unittest.TestCase):
    def test_workspace_header_for_user_scoped_keys(self):
        import os
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-usr-test", "ANTHROPIC_WORKSPACE_ID": "wrkspc_1"}), \
                patch.object(aa, "_client", None):
            client = aa._get_client()
            self.assertEqual(client.default_headers.get("anthropic-workspace-id"), "wrkspc_1")
        self.assertIn("ANTHROPIC_WORKSPACE_ID", aa._describe(bad_request("must include the anthropic-workspace-id header")))
        self.assertEqual(aa._describe(bad_request("other")), "other")


class AccountingTests(unittest.TestCase):
    def test_cost_reconciles_with_tariff(self):
        raw = aa.usage_payload("claude-opus-5-5", {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
                                                   "cache_creation_input_tokens": 1_000_000, "output_tokens": 1_000_000})
        self.assertAlmostEqual(raw["cost"], 4 + 0.2 + 5 + 20)
        price = asyncio.run(accounting.pricing("claude-opus-5-5", "Anthropic", raw["prompt_tokens"], raw))
        usage = accounting.initialize({})
        entry = accounting.record(usage, raw, model="claude-opus-5-5", provider="Anthropic", price=price)
        self.assertAlmostEqual(entry["estimate"]["residual"], 0.0, places=9)
        self.assertEqual(usage["cache_read"], 1_000_000)

    def test_one_hour_writes_and_long_haiku_prompts(self):
        raw = aa.usage_payload("claude-sonnet-5-5", {"input_tokens": 0, "cache_creation_input_tokens": 1_000_000,
                                                     "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000}})
        self.assertAlmostEqual(raw["cost"], 4.0)
        self.assertAlmostEqual(aa.rates("claude-haiku-5-5", 200_000)["prompt"], 0.5e-6)
        self.assertIsNone(aa.rates("claude-unknown-9"))
        self.assertNotIn("cost", aa.usage_payload("claude-unknown-9", {"input_tokens": 5}))


if __name__ == "__main__":
    unittest.main()
