"""Read-only restart verification; prints no secrets or conversation content."""
import json
import os
import sys
import httpx
from .config import BACKEND_BUILD


def main():
    port = int(os.environ.get('CTXAGENT_PORT', '8765'))
    try:
        response = httpx.get(f'http://127.0.0.1:{port}/api/health', timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f'Backend unavailable: {type(exc).__name__}: {exc}')
        return 1
    summary = {k: data.get(k) for k in ('backend_build','server_pid','server_started_at',
               'loaded_source_fingerprint','restart_required','accounting_version')}
    summary['tool_count'] = len(data.get('tools', []))
    print(json.dumps(summary, indent=2))
    current = data.get('backend_build') == BACKEND_BUILD and data.get('accounting_version') == 2 and not data.get('restart_required', True)
    print('CURRENT BACKEND' if current else 'RESTART REQUIRED: stop the existing Python server, then launch again.')
    return 0 if current else 1


if __name__ == '__main__':
    sys.exit(main())
