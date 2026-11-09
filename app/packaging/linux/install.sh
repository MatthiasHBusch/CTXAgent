#!/usr/bin/env bash
# Install or update CTXAgent for the current user (no root needed).
# Run it from the unpacked tarball:
#
#   ./install.sh [--desktop-icon]
#
#   app        ~/.local/opt/CTXAgent            (replaced on update)
#   command    ~/.local/bin/ctxagent
#   app menu   ~/.local/share/applications/ctxagent.desktop
#   data       ~/.local/share/ctxagent          (chats, settings, logs; never touched here)
#
#   --desktop-icon   also put a launcher on the desktop
#
# CTXAGENT_PREFIX overrides the app folder. Remove with ~/.local/opt/CTXAgent/uninstall.sh.
#
# The app was called PiAgent before: install.sh removes a PiAgent install (folder,
# command, menu entry) and the app moves ~/.local/share/piagent to
# ~/.local/share/ctxagent on its first start.
set -euo pipefail

DESKTOP_ICON=0
for arg in "$@"; do
  case "$arg" in
    --desktop-icon) DESKTOP_ICON=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/CTXAgent"
if [ ! -x "$SRC/CTXAgent" ]; then
  echo "Run install.sh from the unpacked CTXAgent folder (CTXAgent/CTXAgent is missing)." >&2
  exit 1
fi
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
PREFIX="${CTXAGENT_PREFIX:-${PIAGENT_PREFIX:-$HOME/.local/opt/CTXAgent}}"
BIN="$HOME/.local/bin"
APPS="$DATA_HOME/applications"
ENTRY="$APPS/ctxagent.desktop"

if pgrep -x -u "$(id -u)" "CTXAgent|ctxagent|PiAgent|piagent" >/dev/null 2>&1; then
  echo "CTXAgent (or PiAgent) is running. Close its window first, then run install.sh again." >&2
  exit 1
fi

VERSION=$("$SRC/CTXAgent" --version 2>/dev/null || echo "CTXAgent")
echo "Installing $VERSION to $PREFIX"
mkdir -p "$(dirname "$PREFIX")"
rm -rf "$PREFIX.new" "$PREFIX.old"
cp -a "$SRC" "$PREFIX.new"
install -m 755 "$HERE/uninstall.sh" "$PREFIX.new/uninstall.sh"
if [ -e "$PREFIX" ]; then
  mv "$PREFIX" "$PREFIX.old"
fi
mv "$PREFIX.new" "$PREFIX"
rm -rf "$PREFIX.old"

# The same app under its former name: replaced by this one.
OLD="$HOME/.local/opt/PiAgent"
if [ -d "$OLD" ] && [ "$OLD" != "$PREFIX" ]; then
  rm -rf "$OLD"
  echo "Removed the former PiAgent install ($OLD); your data moves on the first start."
fi
if [ -L "$BIN/piagent" ]; then
  rm -f "$BIN/piagent"
fi
rm -f "$APPS/piagent.desktop"

mkdir -p "$BIN"
ln -sfn "$PREFIX/CTXAgent" "$BIN/ctxagent"

mkdir -p "$APPS"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=CTXAgent
GenericName=AI agent
Comment=Chat with AI models that read and work on your files
Exec="$PREFIX/CTXAgent"
Icon=$PREFIX/_internal/ctxagent.png
Terminal=false
Categories=Development;Utility;
Keywords=AI;LLM;agent;chat;
StartupNotify=true
StartupWMClass=ctxagent
EOF
chmod 644 "$ENTRY"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS" >/dev/null 2>&1 || true
fi

if [ "$DESKTOP_ICON" = 1 ]; then
  DESKTOP=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
  if [ -d "$DESKTOP" ]; then
    rm -f "$DESKTOP/piagent.desktop"
    install -m 755 "$ENTRY" "$DESKTOP/ctxagent.desktop"
    # GNOME asks before it runs an untrusted launcher; this one is the user's own.
    gio set "$DESKTOP/ctxagent.desktop" metadata::trusted true >/dev/null 2>&1 || true
  fi
fi

# The window (Qt WebEngine) uses a few system libraries; without them CTXAgent
# opens in the web browser instead. Say which ones are missing.
QT="$PREFIX/_internal/PyQt6/Qt6"
missing=$(for f in "$QT/lib/libQt6WebEngineCore.so.6" "$QT/libexec/QtWebEngineProcess" \
                   "$QT"/plugins/platforms/libqxcb.so "$QT"/plugins/platforms/libqwayland*.so; do
            [ -e "$f" ] && LD_LIBRARY_PATH="$PREFIX/_internal:$QT/lib" ldd "$f" 2>/dev/null
          done | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')

echo
echo "Done. Start CTXAgent from the app menu or with: ctxagent"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "(~/.local/bin is not on your PATH; the full command is $PREFIX/CTXAgent)" ;;
esac
if [ -n "$missing" ]; then
  echo
  echo "Missing system libraries for the CTXAgent window: $missing"
  echo "Until they are installed CTXAgent opens in your web browser. For NSS (libnss3, libnspr4, ...):"
  echo "  Ubuntu/Debian/Mint: sudo apt install libnss3     Fedora/Arch: nss     openSUSE: mozilla-nss"
  echo "Other libraries: install the package your distribution names for them (README.txt)."
fi
