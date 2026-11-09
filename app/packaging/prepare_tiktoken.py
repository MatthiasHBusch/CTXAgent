"""Fill a tiktoken cache folder for the bundle, so token counting works offline.

    python packaging/prepare_tiktoken.py build/tiktoken_cache

tiktoken downloads encodings on first use and caches them under the SHA-1 of
the download URL. The frozen app points TIKTOKEN_CACHE_DIR at the bundled copy
(paths.prepare_runtime). Reuses a copy from the default cache when there is
one; tiktoken verifies the file's hash either way.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

ENCODINGS = {  # what ctxagent/tokens.py uses
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
}


def main(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    default_cache = Path(os.environ.get("DATA_GYM_CACHE_DIR")
                         or Path(tempfile.gettempdir()) / "data-gym-cache")
    for name, url in ENCODINGS.items():
        key = hashlib.sha1(url.encode()).hexdigest()
        if not (target / key).exists() and (default_cache / key).is_file():
            shutil.copy2(default_cache / key, target / key)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(target)
    import tiktoken
    for name, url in ENCODINGS.items():
        enc = tiktoken.get_encoding(name)
        assert enc.encode("hello world") == [15339, 1917], name
        key = hashlib.sha1(url.encode()).hexdigest()
        print(f"{name}: {target / key} ({(target / key).stat().st_size} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
