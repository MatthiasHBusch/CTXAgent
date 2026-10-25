"""Run all offline suites without writing to the user's chat database/blob store."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parent.parent
    failed = []
    with tempfile.TemporaryDirectory(prefix='ctxagent-tests-') as data:
        env = {**os.environ, 'CTXAGENT_DATA': data, 'PYTHONIOENCODING': 'utf-8',
               'PYTHONDONTWRITEBYTECODE': '1'}
        for module in ('tests.test_compaction', 'tests.test_readers', 'tests.test_productivity', 'tests.test_cache_accounting', 'tests.test_workflows', 'tests.test_cadence', 'tests.test_research', 'tests.test_flex', 'tests.test_chat_fixes', 'tests.test_context_report', 'tests.test_freeze_review', 'tests.test_workspace', 'tests.test_document_edits', 'tests.test_document_catalog', 'tests.test_agents', 'tests.test_browsing', 'tests.test_browser_workspace', 'tests.test_browser_redirects', 'tests.test_chat_hierarchy', 'tests.test_markdown', 'tests.test_anthropic_api', 'tests.test_seqdep', 'tests.test_bench_phase0'):
            print(f'\n=== {module} ===', flush=True)
            result = subprocess.run([sys.executable, '-B', '-m', module], cwd=root, env=env)
            if result.returncode:
                failed.append(module)
                import shutil
                destination = root / '.ctxagent' / 'failed-test-data' / module.replace('.', '_')
                shutil.copytree(data, destination, dirs_exist_ok=True)
    print('\n' + ('FAILED: ' + ', '.join(failed) if failed else 'All offline suites passed.'))
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
