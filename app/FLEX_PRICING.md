# Flex pricing and service-tier routing

## Activation

Restart the Python backend, hard-refresh the browser, then run:

```powershell
python -B -m ctxagent.check_backend
```

Expected: `CURRENT BACKEND`, `backend_build: flex-v1`, `accounting_version: 2`,
and `restart_required: false`.

In Settings, choose **Pricing / service tier** separately for the main model and
the compaction/summary model, then save settings. Existing chats remain on
standard routing unless explicitly changed.

| Mode | Behaviour |
| --- | --- |
| `standard` | Normal provider routing. No Flex discovery for generation and no `service_tier` opt-in. |
| `prefer_flex` | Discover eligible Flex endpoints and try them; allow standard-price fallback if none are found or on an eligible pre-stream HTTP rejection. |
| `flex_only` | Require discovered Flex endpoints. Stop rather than fall back to standard routing. |

Configuration keys: `pricing_mode` and `compaction.summary_pricing_mode`.

**Refresh Flex availability** fetches provider metadata, not a test generation.
It shows advertised input/output tariffs and savings against the matching standard
endpoint, respecting provider-region policy and advertised tool-use support.
Discounts are not hard-coded to 50%; availability and prices can change. The preview
uses an estimated prompt size for long-context pricing thresholds.

## Billing and retry boundaries

- Flex requests restrict `provider.only` to explicit `/flex` endpoint tags rather
  than relying on `service_tier=flex` alone. Gateway fallback stays within that set.
- Standard fallback is allowed only in `prefer_flex`, on HTTP 404, 408, 429, 502,
  503 or 504 before a successful stream starts, or a narrowly recognized
  tier-unsupported HTTP 400. An HTTP 408 rejection is distinct from a local/network timeout.
- Network timeouts, authentication/credit failures, unknown execution failures,
  malformed responses and partial generations are not replayed for tier fallback.
- Returned usage/cost is authoritative. The provider-reported tier is recorded;
  missing tier information is not treated as confirmation of Flex.
- A non-Flex reported tier despite Flex routing produces a warning, not a repeat
  request. `flex_only` controls routing; it cannot override a provider's billing report.
- Usage included in an SSE error chunk is retained before the error is surfaced.
- Endpoint tariff estimates remain separate from actual reported cost. Input,
  cache-read/write, visible-output and reasoning cost categories retain their
  existing estimate/unallocated labels.
- The existing summary completion reasoning ladder is separate: a completed but
  empty answer can trigger a lower-effort attempt. Each reported usage is accounted
  for; this is not a free retry or a tier fallback.

## Verification completed

Offline Flex regressions cover routing modes, discovery, regional restrictions,
pre-stream fallback boundaries, no replay after partial/network failures, reported
tier mismatches and SSE-error usage accounting. Mocked Playwright coverage checks
both model selectors; screenshots were visually inspected.

One successful live Astra billing test reported `service_tier: flex`, 15 input
and 5 output tokens, with a provider-reported charge of **$0.000200**. At the
matching standard endpoint's advertised tariffs this would be **$0.000400**.
This validates that request, not future availability or every model's discount.
The paid test was not repeated during final documentation.

Offline verification commands:

```powershell
python -B -m tests.run_all
node --check web/app.js
python -B -m tests.flex_ui_smoke
```

`tests.flex_smoke` is a live test and can incur API charges; it is not part of the
offline runner.
