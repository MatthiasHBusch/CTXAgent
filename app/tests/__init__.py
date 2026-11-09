import os

# The offline suites must never read or write the real OS credential store:
# keys come from the environment only unless a test installs its own backend.
os.environ.setdefault("CTXAGENT_KEYSTORE", "env")
