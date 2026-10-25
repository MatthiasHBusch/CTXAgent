# External sources and toolchains for the benchmark

Nothing in this directory is redistributed; `fetch.ps1` (Windows) or `fetch.sh` (Linux/macOS)
fetches and pins everything the benchmark needs. Each item is optional in the sense that a
session family only needs its own sources (Python exercises need nothing but Python).

| path | what | pinned at | needed by |
|---|---|---|---|
| `polyglot-benchmark/` | Aider polyglot benchmark (Exercism exercises in six languages) | commit `7e0611e` | `poly:`, `polydep:`, `mixdep:`, `marathon:` |
| `SciCode/` | SciCode benchmark (problems, test cases, `eval/data/test_data.h5`) | commit `e3158ea` | `sci:`, `mixdep:`, `marathon:` |
| `go/` | portable Go toolchain (no installer; `bench/tasks_polyglot.py` sets GOCACHE/GOPATH per trial) | go 1.27.3 | Go track of `marathon:` |
| `js_shared/node_modules/` | the JavaScript track's dev dependencies (jest, babel), shared by every exercise through a directory junction / symlink | `js_shared.package.json` | JavaScript track of `marathon:` |
| `opencode/` | OpenCode CLI, the open-source harness of the harness comparison | `opencode-ai` 1.19.3 (`opencode.package.json`) | `--harnesses opencode` |
| `cc/` | pinned Claude Code CLI | `@anthropic-ai/claude-code` 2.1.301 | `--harnesses claude_code` |
| Codex CLI | installed globally: `npm i -g @openai/codex@0.163.0` | 0.163.0 | `--harnesses codex` |

SciCode's `test_data.h5` is distributed by the SciCode authors separately from their repository;
`fetch.*` prints the instruction from their README if the file is missing after the clone.
Run the checker oracles after fetching: `python -m bench.tasks_polyglot 5 0 python`,
`python -m bench.tasks_scicode` (every dev-split step must pass with the reference solution).
