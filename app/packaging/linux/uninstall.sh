#!/usr/bin/env bash
# Remove CTXAgent for the current user.
#
#   ~/.local/opt/CTXAgent/uninstall.sh [--purge]
#
# Removes the app, the `ctxagent` command and the menu entry. Your chats,
# settings and logs (~/.local/share/ctxagent) stay unless you pass --purge,
# which also deletes them, ~/.config/ctxagent, ~/.cache/ctxagent and the API keys
# CTXAgent stored in the system keyring. A folder chosen with CTXAGENT_DATA is never deleted.
set -euo pipefail

PURGE=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="${CTXAGENT_PREFIX:-${PIAGENT_PREFIX:-$HOME/.local/opt/CTXAgent}}"
BIN="$HOME/.local/bin"

if pgrep -x -u "$(id -u)" "CTXAgent|ctxagent" >/dev/null 2>&1; then
  echo "CTXAgent is running. Close its window first, then run uninstall.sh again." >&2
  exit 1
fi

if [ "$PURGE" = 1 ] && [ -x "$PREFIX/CTXAgent" ]; then
  env -u CTXAGENT_DATA "$PREFIX/CTXAgent" --run-module ctxagent.keys --forget-all \
    || echo "Could not remove the stored API keys (see your keyring app)." >&2
fi

if [ -L "$BIN/ctxagent" ] && [ "$(readlink "$BIN/ctxagent")" = "$PREFIX/CTXAgent" ]; then
  rm -f "$BIN/ctxagent"
fi
rm -f "$DATA_HOME/applications/ctxagent.desktop"
DESKTOP=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
rm -f "$DESKTOP/ctxagent.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true
fi
rm -rf "$PREFIX"
echo "CTXAgent removed from $PREFIX."

if [ "$PURGE" = 1 ]; then
  rm -rf "$DATA_HOME/ctxagent" "$CONFIG_HOME/ctxagent" "$CACHE_HOME/ctxagent"
  # Left by the app under its former name (PiAgent), if any.
  rm -rf "$DATA_HOME/piagent" "$CONFIG_HOME/piagent" "$CACHE_HOME/piagent"
  echo "Data removed: $DATA_HOME/ctxagent, $CONFIG_HOME/ctxagent, $CACHE_HOME/ctxagent."
else
  echo "Your data stays in $DATA_HOME/ctxagent (delete it with --purge)."
fi
