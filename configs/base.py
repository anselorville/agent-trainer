class _DummyConfig:
    def __init__(self):
        self.settings = {}

    def set(self, key, value):
        self.settings[key] = value


class BaseApplication:
    """
    Minimal stub compatible with AgentLightning's GunicornApp usage.

    If gunicorn is actually invoked, raise a clear error explaining Windows
    incompatibility.
    """

    def __init__(self, *args, **kwargs):
        self.cfg = _DummyConfig()

    def load_config(self):
        return None

    def load(self):
        raise RuntimeError("Gunicorn is not supported on Windows in this environment.")

    def run(self):
        raise RuntimeError("Gunicorn is not supported on Windows in this environment.")
