CTXAgent for Linux (x86_64)
==========================

Runs on Ubuntu 22.04, Debian 12, Fedora 36, Linux Mint 21 or newer (glibc 2.35+),
in an X11 or Wayland desktop session.

Install or update (no root needed; your chats and settings are kept):

    tar xzf CTXAgent-<version>-linux-x86_64.tar.gz
    cd CTXAgent-<version>-linux-x86_64
    ./install.sh                  # add --desktop-icon for a launcher on the desktop

Then start CTXAgent from the app menu, or run `ctxagent` in a terminal.
Coming from PiAgent (the former name): install.sh replaces it, and the first
start moves your chats and settings from ~/.local/share/piagent.
The first start asks for an API key (Settings > API keys).

The window uses the desktop's own graphics, sound and font libraries and NSS.
install.sh names any library that is missing; without them CTXAgent opens in
your web browser instead. NSS is the one a desktop may lack:

    Ubuntu, Debian, Mint:  sudo apt install libnss3
    Fedora:                sudo dnf install nss
    Arch, Manjaro:         sudo pacman -S nss
    openSUSE:              sudo zypper install mozilla-nss

Where things are:

    app        ~/.local/opt/CTXAgent
    data       ~/.local/share/ctxagent      chats, settings, logs (logs/ctxagent.log)
    API keys   the system keyring (GNOME Keyring / KWallet). Without a running
               keyring they go to ~/.local/share/ctxagent/credentials.json,
               readable only by you; Settings shows which one is used.

If the window stays empty or does not open, `ctxagent --browser` shows the same
app in your web browser; ~/.local/share/ctxagent/logs/ctxagent.log says why.

Uninstall:  ~/.local/opt/CTXAgent/uninstall.sh   (add --purge to delete your data and keys too)

More: https://github.com/MatthiasHBusch/CTXAgent
