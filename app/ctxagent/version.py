"""Product identity: the one place that names and numbers the app.

The Python package stays `ctxagent`; everything a user sees (window title,
installer, data folder, shortcuts) uses APP_NAME. packaging/ reads this file.
"""

APP_NAME = "CTXAgent"
# Former names: their data folder and stored keys are taken over on first start
# (paths.data_dir, keys.KeyringBackend).
LEGACY_APP_NAMES = ("PiAgent",)
__version__ = "0.1.0"
PUBLISHER = "Matthias Busch"
# Windows taskbar grouping and the single-instance identity.
APP_USER_MODEL_ID = "MatthiasBusch.CTXAgent"
