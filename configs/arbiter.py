class Arbiter:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def run(self):
        raise RuntimeError("Gunicorn is not supported on Windows in this environment.")
