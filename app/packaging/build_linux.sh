#!/usr/bin/env bash
# Build the CTXAgent Linux bundle: venv -> pip install -> PyInstaller -> self-test -> tar.gz.
#
#   packaging/build_linux.sh [--python PY] [--clean] [--skip-tarball]
#
# Output:  dist/CTXAgent/                                    the onedir bundle
#          dist/linux/CTXAgent-<version>-linux-x86_64.tar.gz  bundle + install.sh
#
# The build venv is .venv-linux in the checkout (git-ignored); nothing is
# installed into the Python that creates it. Dependencies are pinned by
# packaging/constraints.txt and packaging/constraints-linux.txt (window, key
# store). The interpreter needs the venv module and a shared libpython
# (python.org, distribution, deadsnakes and conda builds have one); 3.11 is
# what the Windows build uses. Building needs neither root nor system packages.
# The bundle runs on the glibc of the build host or newer: build on the oldest
# distribution you want to support. Running the window needs libnss3 and a few
# X11 libraries (DESKTOP_APP.md, section Linux).
#
#   --python PY      interpreter for the venv (default: python3.11, else python3)
#   --clean          recreate the venv and wipe build/ and dist/ first
#   --skip-tarball   stop after the bundle and its self-test
set -euo pipefail

PYTHON=""
CLEAN=0
TARBALL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --skip-tarball) TARBALL=0; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [ -z "$PYTHON" ]; then
  PYTHON=$(command -v python3.11 || command -v python3 || true)
fi
if [ -z "$PYTHON" ]; then
  echo "no python3 found; pass --python /path/to/python3.11" >&2
  exit 1
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VENV="$ROOT/.venv-linux"
PY="$VENV/bin/python"

step() { printf '\033[36m==> %s\033[0m\n' "$*"; }

if [ "$CLEAN" = 1 ]; then
  rm -rf "$VENV" build dist
fi
if [ ! -x "$PY" ]; then
  step "create venv ($PYTHON)"
  "$PYTHON" -m venv "$VENV"
fi
step "upgrade pip"
"$PY" -m pip install --disable-pip-version-check -q --upgrade pip
step "install dependencies"
"$PY" -m pip install --disable-pip-version-check -r requirements.txt \
  -c packaging/constraints.txt -c packaging/constraints-linux.txt "pywebview[qt]" pyinstaller

VERSION=$("$PY" -c "from ctxagent.version import __version__; print(__version__)")
printf '\033[32mCTXAgent %s\033[0m\n' "$VERSION"

if [ ! -f packaging/ctxagent.png ]; then
  step "draw icon"
  "$PY" packaging/make_icon.py
fi
step "bundle tokenizer data"
"$PY" packaging/prepare_tiktoken.py build/tiktoken_cache
step "PyInstaller"
"$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/ctxagent.spec

step "bundle self-test"
dist/CTXAgent/CTXAgentCLI --run-module ctxagent.selftest

echo "bundle: dist/CTXAgent, $(find dist/CTXAgent -type f | wc -l) files, $(du -sm dist/CTXAgent | cut -f1) MB"
if [ "$TARBALL" = 0 ]; then
  exit 0
fi

NAME="CTXAgent-$VERSION-linux-$(uname -m)"
STAGE="dist/linux/$NAME"
step "tarball"
rm -rf "$STAGE" "dist/linux/$NAME.tar.gz"
mkdir -p "$STAGE"
cp -a dist/CTXAgent "$STAGE/CTXAgent"
install -m 755 packaging/linux/install.sh packaging/linux/uninstall.sh "$STAGE/"
install -m 644 packaging/linux/README.txt "$STAGE/"
# Owner root:root in the headers: nothing of the build account (user name, ids) goes out.
tar -C dist/linux --owner=0 --group=0 --numeric-owner -cf - "$NAME" | gzip -9 > "dist/linux/$NAME.tar.gz"
rm -rf "$STAGE"
printf '\033[32mtarball: %s (%s MB)\033[0m\n' "dist/linux/$NAME.tar.gz" "$(du -m "dist/linux/$NAME.tar.gz" | cut -f1)"
