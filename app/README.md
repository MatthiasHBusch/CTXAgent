# CTXAgent desktop app

CTXAgent is a local LLM agent with a GUI (chats, file and shell tools, document readers,
browsing, sub-agents) whose **context is managed by the model itself**: it compacts finished
work into hand-off summaries it writes and drops tool output it has already used, with
`recall` to get anything back. This folder holds the desktop app built on that agent: the
source, the installer builds and their documentation.

The app is developed further than the version the paper evaluated. That version stays frozen
in the repository root (`../ctxagent`, `../bench`, `../web`); the results in the paper refer to
it, not to this folder.

## Download and install

Installers are attached to the GitHub release
[**app-v0.2.0**](https://github.com/MatthiasHBusch/CTXAgent/releases/tag/app-v0.2.0), together
with `SHA256SUMS.txt`.

**Windows 10/11 (x64):** `CTXAgent-Setup-0.2.0.exe`

- Installs for the current user, no administrator rights needed
  (`%LOCALAPPDATA%\Programs\CTXAgent`, Start-menu entry, optional desktop icon). Uninstall in
  *Settings > Apps*.
- The installer is not code-signed, so SmartScreen may warn: *More info > Run anyway*.
- The window uses Microsoft Edge WebView2, which Windows 11 includes. Without it the app opens
  in your web browser instead.

**Linux (x86_64, glibc 2.35+: Ubuntu 22.04, Debian 12, Fedora 36, Mint 21 or newer):**
`CTXAgent-0.2.0-linux-x86_64.tar.gz`

```bash
tar xzf CTXAgent-0.2.0-linux-x86_64.tar.gz
cd CTXAgent-0.2.0-linux-x86_64
./install.sh                  # no root; add --desktop-icon for a launcher on the desktop
```

Start CTXAgent from the app menu or run `ctxagent`. The window needs NSS
(`sudo apt install libnss3` on Ubuntu/Debian); `install.sh` names anything missing, and
`ctxagent --browser` opens the same app in your web browser.
[`packaging/linux/README.txt`](packaging/linux/README.txt) has the details and the uninstaller.

Check a download: `sha256sum -c SHA256SUMS.txt --ignore-missing` (Linux) or
`Get-FileHash CTXAgent-Setup-0.2.0.exe` (PowerShell) against `SHA256SUMS.txt`.

## First steps

1. **API key.** The first start asks for one (*Settings > API keys*): an
   [OpenRouter](https://openrouter.ai/keys) key for models of every provider, or an
   [Anthropic](https://platform.claude.com/settings/keys) key for Claude through the direct
   API. Keys are kept in the system's credential store (Windows Credential Manager, the
   GNOME Keyring or KWallet on Linux). You pay the provider for what the agent uses; the cost
   of each request is shown in the chat.
2. **Working directory.** Add a folder in the **Tools** tab before asking the agent to touch
   files. File and shell tools are confined to the folders you list.
3. **Chat settings.** A new chat starts with Claude Opus 5.5 at reasoning effort xhigh, a
   700k context window, unlimited output, self-managed context with deep compaction as a
   safety net at 90 % of the window, and every tool on. Settings save as you change them;
   while the agent works they apply at its next model request.

Chats, settings and logs stay on your computer: `%LOCALAPPDATA%\CTXAgent` on Windows,
`~/.local/share/ctxagent` on Linux (`logs/ctxagent.log` there if something goes wrong).
[DESKTOP_APP.md](DESKTOP_APP.md) describes the app, its command-line options, data locations
and known limits; [SETTINGS.md](SETTINGS.md) the settings.

## Run or build from source

Python 3.11 or newer:

```bash
cd app
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ctxagent                   # server and GUI at http://127.0.0.1:8765 in your browser
python -m ctxagent.desktop           # native window (needs pywebview on Windows, PyQt6 on Linux)
```

Instead of the key dialog you can put `OPENROUTER_API_KEY=...` or `ANTHROPIC_API_KEY=...` in
`app/.env` (see [`.env.example`](.env.example)). From a source checkout, chats and settings
live in `app/.ctxagent/`.

Installers:

- Windows: `powershell -ExecutionPolicy Bypass -File packaging\build.ps1` (PyInstaller bundle,
  self-test, then [Inno Setup 6](https://jrsoftware.org/isinfo.php)).
- Linux: `bash packaging/build_linux.sh` (bundle, self-test, tarball; build on the oldest
  distribution you want to support).

Versions are pinned in `packaging/constraints.txt` and `packaging/constraints-linux.txt`.
Tests: `python -m tests.run_all`.

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md): how the agent and its compaction engine work, the tools,
  document readers and the MCP server for other agents, the layout of the code.
- [DESKTOP_APP.md](DESKTOP_APP.md), [SETTINGS.md](SETTINGS.md),
  [INSIGHTS.md](INSIGHTS.md) (usage insights and the report),
  [DOCUMENT_WORKSPACE.md](DOCUMENT_WORKSPACE.md),
  [CANVAS_AND_SCRATCHPAD.md](CANVAS_AND_SCRATCHPAD.md),
  [AGENTS_AND_BROWSING.md](AGENTS_AND_BROWSING.md), [FLEX_PRICING.md](FLEX_PRICING.md).
- Design notes: [RESEARCH.md](RESEARCH.md), [MODELS.md](MODELS.md),
  [FREEZE_REVIEW.md](FREEZE_REVIEW.md), [COMPACTION_CADENCE.md](COMPACTION_CADENCE.md),
  [CACHE_AUDIT.md](CACHE_AUDIT.md), [EFFICIENCY_AUDIT.md](EFFICIENCY_AUDIT.md).

## License

The app in this folder is released under the
[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only). Copyright (C) 2026
Matthias Busch. The rest of the repository (the paper's agent version, the benchmark and the
results) stays under CC BY 4.0, see [`../LICENSE`](../LICENSE).

The installers bundle a Python runtime and third-party packages under their own licenses,
among them PyMuPDF (AGPL-3.0) and, in the Linux build, PyQt6 (GPL-3.0) and Qt (LGPL-3.0);
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) lists them. The vendored JavaScript in
`web/vendor/` has its license next to it.

If you use CTXAgent in research, please cite the paper (see [`../CITATION.cff`](../CITATION.cff)).
