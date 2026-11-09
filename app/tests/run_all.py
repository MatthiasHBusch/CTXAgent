"""Run all offline suites without writing to the user's chat database/blob store."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile


MODULES = (
    'tests.test_compaction',
    'tests.test_readers',
    'tests.test_productivity',
    'tests.test_cache_accounting',
    'tests.test_workflows',
    'tests.test_cadence',
    'tests.test_research',
    'tests.test_flex',
    'tests.test_chat_fixes',
    'tests.test_context_report',
    'tests.test_freeze_review',
    'tests.test_workspace',
    'tests.test_document_edits',
    'tests.test_document_catalog',
    'tests.test_agents',
    'tests.test_browsing',
    'tests.test_browser_workspace',
    'tests.test_browser_redirects',
    'tests.test_chat_hierarchy',
    'tests.test_markdown',
    'tests.test_anthropic_api',
    'tests.test_seqdep',
    'tests.test_bench_phase0',
    'tests.test_settings',
    'tests.test_tool_call_failures',
    'tests.test_desktop',
    'tests.test_guidance',
    'tests.test_insights',
    'tests.test_queue',
    'tests.test_scratchpad',
    'tests.test_canvas',
    'tests.test_dynamic_tools',
    'tests.test_view_pairing',
    'tests.test_token_calibration',
    'tests.test_new_chat_defaults',
)


#: Suites of the benchmark (bench/), skipped where it is not part of the checkout (the
#: app-only tree of the public repository).
NEEDS_BENCH = {'tests.test_seqdep', 'tests.test_bench_phase0'}


def main():
    root = Path(__file__).resolve().parent.parent
    failed = []
    has_bench = (root / 'bench' / '__init__.py').is_file()
    with tempfile.TemporaryDirectory(prefix='ctxagent-tests-') as data:
        env = {**os.environ, 'CTXAGENT_DATA': data, 'PYTHONIOENCODING': 'utf-8',
               'PYTHONDONTWRITEBYTECODE': '1'}
        for module in MODULES:
            if module in NEEDS_BENCH and not has_bench:
                print(f'\n=== {module} === skipped (no bench/ in this checkout)', flush=True)
                continue
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
