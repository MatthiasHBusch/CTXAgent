# Third-party components of the CTXAgent installers

The installers (`CTXAgent-Setup-<version>.exe` for Windows, `CTXAgent-<version>-linux-x86_64.tar.gz`)
are PyInstaller bundles. Besides the app (AGPL-3.0, [LICENSE](LICENSE)) they contain the
components below, each under its own license. The license texts ship inside the packages;
the projects' pages have them as well. The list was generated from the modules found in the
bundles of release app-v0.1.0 and the package metadata of the build environments.

## Runtime and native parts

| Component | Build | License |
|---|---|---|
| Python 3.11 (Windows), 3.12 (Linux) | both | PSF-2.0 |
| PyInstaller bootloader 6.22 | both | GPL-2.0-or-later with the bootloader exception (permits bundling) |
| Microsoft Visual C++ runtime (`VCRUNTIME140*.dll`, `MSVCP140.dll`), Universal CRT forwarders | Windows | Microsoft redistributable runtime |
| Microsoft Edge WebView2 SDK (`Microsoft.Web.WebView2.*.dll`, `WebView2Loader.dll`, shipped with pywebview) | Windows | Microsoft WebView2 SDK license (BSD-style, redistributable); the WebView2 runtime itself is not bundled |
| Qt 6 and Qt WebEngine (Chromium) libraries (via PyQt6-Qt6, PyQt6-WebEngine-Qt6) | Linux | LGPL-3.0; Chromium parts under BSD-3-Clause and further licenses listed by Qt |
| tiktoken encoding `cl100k_base` (token counting) | both | MIT (tiktoken) |
| markdown-it 15.0.2 (`web/vendor/`, in the app source) | both | MIT |

## Python packages

| Package | Version | Build | License |
|---|---|---|---|
| annotated-doc | 0.0.5 | both | MIT |
| annotated-types | 0.8.0 | both | MIT |
| anthropic | 1.17.0 | both | MIT |
| anyio | 4.16.0 | both | MIT |
| attrs | 26.1.0 | both | MIT |
| backports.tarfile | 1.2.0 | Windows | MIT |
| beautifulsoup4 | 4.12.2 | both | MIT |
| bottle | 0.13.4 | both | MIT |
| certifi | 2026.10.23 | both | MPL-2.0 |
| cffi | 2.1.1 | both | MIT-0 |
| chardet | 5.2.0 | both | LGPL-2.1-or-later |
| charset-normalizer | 3.5.2 | both | MIT |
| click | 8.5.0 | both | BSD-3-Clause |
| clr_loader | 0.3.1 | Windows | MIT |
| cryptography | 50.0.2 | both | Apache-2.0 OR BSD-3-Clause |
| docstring_parser | 0.18.0 | both | MIT |
| et_xmlfile | 2.0.0 | both | MIT |
| fastapi | 0.123.10 | both | MIT |
| h11 | 0.16.0 | both | MIT |
| httpcore | 1.0.9 | both | BSD-3-Clause |
| httpcore2 | 2.16.0 | both | BSD-3-Clause |
| httptools | 0.9.0 | both | MIT |
| httpx | 0.28.1 | both | BSD-3-Clause |
| httpx-sse | 0.4.3 | both | MIT |
| httpx2 | 2.16.0 | both | BSD-3-Clause |
| idna | 3.20 | both | BSD-3-Clause |
| importlib_metadata | 10.1.0 | Windows | Apache-2.0 |
| jaraco.classes | 3.4.0 | both | MIT |
| jaraco.context | 6.1.2 | both | MIT |
| jaraco.functools | 4.6.0 | both | MIT |
| jeepney | 0.9.0 | Linux | MIT |
| jiter | 0.18.0 | both | MIT |
| jsonschema | 4.25.1 | both | MIT |
| jsonschema-specifications | 2025.9.1 | both | MIT |
| keyring | 25.7.0 | both | MIT |
| lxml | 4.9.3 | both | BSD-3-Clause |
| mcp | 1.25.0 | both | MIT |
| more-itertools | 11.2.1 | both | MIT |
| numpy | 1.26.4 | both | BSD-3-Clause |
| openpyxl | 3.1.5 | both | MIT |
| opentelemetry-api | 1.45.2 | both | Apache-2.0 |
| packaging | 26.3 | both | Apache-2.0 OR BSD-2-Clause |
| pandas | 2.1.4 | both | BSD-3-Clause |
| pillow | 10.4.0 | both | HPND (Historical Permission Notice and Disclaimer (HPND)) |
| proxy_tools | 0.1.0 | both | MIT |
| psutil | 7.2.2 | both | BSD-3-Clause |
| pycparser | 3.12 | both | BSD-3-Clause |
| pydantic | 2.12.5 | both | MIT |
| pydantic-settings | 2.16.2 | both | MIT |
| pydantic_core | 2.41.5 | both | MIT |
| PyMuPDF | 1.27.2.2 | both | AGPL-3.0 (or an Artifex commercial license) |
| PyQt6 | 6.12.0 | Linux | GPL-3.0-only |
| PyQt6-Qt6 | 6.12.0 | Linux | LGPL-3.0 |
| PyQt6-WebEngine | 6.140.0 | Linux | GPL-3.0-only |
| PyQt6-WebEngine-Qt6 | 6.140.0 | Linux | LGPL-3.0 |
| PyQt6_sip | 13.13.0 | Linux | BSD-2-Clause |
| python-dateutil | 4.0.0 | both | BSD-3-Clause |
| python-docx | 1.2.0 | both | MIT |
| python-dotenv | 1.2.4 | both | BSD-3-Clause |
| python-multipart | 0.0.33 | both | Apache-2.0 |
| python-pptx | 1.0.2 | both | MIT |
| pythonnet | 3.2.1 | Windows | MIT |
| pytz | 2026.6 | both | MIT |
| pywebview | 6.2.1 | both | BSD-3-Clause |
| pywin32 | 312 | Windows | PSF-2.0 |
| pywin32-ctypes | 0.2.3 | Windows | BSD-3-Clause |
| PyYAML | 6.0.3 | both | MIT |
| QtPy | 2.4.3 | Linux | MIT |
| referencing | 0.37.0 | both | MIT |
| regex | 2026.9.29 | both | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | both | Apache-2.0 |
| rpds-py | 2026.9.1 | both | MIT |
| SecretStorage | 3.5.0 | Linux | BSD-3-Clause |
| setuptools | 65.5.0 / 84.2.1 | both | MIT |
| sniffio | 1.3.1 | both | MIT OR Apache-2.0 |
| soupsieve | 4.0.1 | both | MIT |
| sse-starlette | 3.6.0 | both | BSD-3-Clause |
| starlette | 0.50.0 | both | BSD-3-Clause |
| tiktoken | 0.9.0 | both | MIT |
| truststore | 0.10.5 | both | MIT |
| typing-inspection | 0.4.5 | both | MIT |
| typing_extensions | 4.16.0 | both | PSF-2.0 |
| tzdata | 2026.6 | Windows | Apache-2.0 |
| urllib3 | 2.8.0 | both | MIT |
| uvicorn | 0.40.0 | both | BSD-3-Clause |
| uvloop | 0.23.0 | Linux | MIT OR Apache-2.0 |
| watchfiles | 1.3.0 | both | MIT |
| websockets | 17.4 | both | BSD-3-Clause |
| xlsxwriter | 3.3.1 | both | BSD-2-Clause |
| zipp | 4.1.2 | Windows | MIT |

Where a version shows two numbers, the first is the Windows build, the second the Linux build.
