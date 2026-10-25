#!/usr/bin/env bash
# Fetch and pin the benchmark's external sources and toolchains (see README.md here).
# Usage: bash bench/external/fetch.sh   (set NO_GO=1 / NO_JS=1 / NO_OPENCODE=1 / NO_CC=1 to skip parts)
set -euo pipefail
cd "$(dirname "$0")"

[ -d polyglot-benchmark ] || git clone https://github.com/Aider-AI/polyglot-benchmark.git
git -C polyglot-benchmark checkout -q 7e0611e

[ -d SciCode ] || git clone https://github.com/scicode-bench/SciCode.git
git -C SciCode checkout -q e3158ea
[ -f SciCode/eval/data/test_data.h5 ] || echo "SciCode/eval/data/test_data.h5 is missing: download it as described in SciCode/README.md (section 'Evaluation')."

if [ -z "${NO_GO:-}" ] && [ ! -x go/bin/go ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) tgz=go1.27.3.linux-amd64.tar.gz ;;
    Linux-aarch64) tgz=go1.27.3.linux-arm64.tar.gz ;;
    Darwin-arm64) tgz=go1.27.3.darwin-arm64.tar.gz ;;
    Darwin-x86_64) tgz=go1.27.3.darwin-amd64.tar.gz ;;
    *) echo "unknown platform, fetch Go 1.27.3 by hand into bench/external/go"; tgz= ;;
  esac
  if [ -n "$tgz" ]; then curl -sSL -o "$tgz" "https://go.dev/dl/$tgz" && tar xzf "$tgz" && rm "$tgz"; fi
fi

if [ -z "${NO_JS:-}" ]; then
  mkdir -p js_shared && cp js_shared.package.json js_shared/package.json
  (cd js_shared && npm install --no-audit --no-fund)
fi

if [ -z "${NO_OPENCODE:-}" ]; then
  mkdir -p opencode && cp opencode.package.json opencode/package.json
  (cd opencode && npm install --no-audit --no-fund)
fi

if [ -z "${NO_CC:-}" ]; then
  mkdir -p cc && (cd cc && { [ -f package.json ] || npm init -y >/dev/null; } && npm install --no-audit --no-fund "@anthropic-ai/claude-code@2.1.301")
fi
echo "done. Codex CLI is used from the global install: npm i -g @openai/codex@0.163.0"
