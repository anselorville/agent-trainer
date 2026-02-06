import sys
import types
import socket
import os

def apply_patches():
    if sys.platform != 'win32':
        return

    # os monkeypatch
    if not hasattr(os, 'geteuid'): os.geteuid = lambda: 0
    if not hasattr(os, 'getegid'): os.getegid = lambda: 0
    if not hasattr(os, 'getuid'): os.getuid = lambda: 0
    if not hasattr(os, 'getgid'): os.getgid = lambda: 0
    if not hasattr(os, 'setuid'): os.setuid = lambda x: None
    if not hasattr(os, 'setgid'): os.setgid = lambda x: None

    # fcntl monkeypatch
    if 'fcntl' not in sys.modules:
        m = types.ModuleType('fcntl')
        sys.modules['fcntl'] = m
        def dummy_fn(*args, **kwargs): return 0
        m.fcntl = dummy_fn
        m.ioctl = dummy_fn
        m.flock = dummy_fn
        m.lockf = dummy_fn

    # pwd monkeypatch
    if 'pwd' not in sys.modules:
        p = types.ModuleType('pwd')
        sys.modules['pwd'] = p
        def getpwuid(uid):
            class StructPasswd:
                pw_name = 'user'
            return StructPasswd()
        p.getpwuid = getpwuid

    # grp monkeypatch
    if 'grp' not in sys.modules:
        g = types.ModuleType('grp')
        sys.modules['grp'] = g
        def getgrgid(gid):
            class StructGroup:
                gr_name = 'group'
            return StructGroup()
        g.getgrgid = getgrgid

    # signal monkeypatch
    import signal
    for sig in "HUP QUIT INT TERM TTIN TTOU USR1 USR2 WINCH CHLD".split():
        sig_name = f"SIG{sig}"
        if not hasattr(signal, sig_name):
            setattr(signal, sig_name, 99)

    # socket.AF_UNIX monkeypatch
    if not hasattr(socket, 'AF_UNIX'):
        socket.AF_UNIX = 1

    # Inline mocks for gunicorn components
    class _DummyConfig:
        def __init__(self): self.settings = {}
        def set(self, key, value): self.settings[key] = value

    class BaseApplication:
        def __init__(self, *args, **kwargs): self.cfg = _DummyConfig()
        def load_config(self): return None
        def load(self): raise RuntimeError("Gunicorn is not supported on Windows.")
        def run(self): raise RuntimeError("Gunicorn is not supported on Windows.")

    class Arbiter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def run(self): raise RuntimeError("Gunicorn is not supported on Windows.")

    # Gunicorn mocks
    gunicorn = types.ModuleType('gunicorn')
    sys.modules['gunicorn'] = gunicorn
    
    g_app = types.ModuleType('gunicorn.app')
    sys.modules['gunicorn.app'] = g_app
    
    g_app_base = types.ModuleType('gunicorn.app.base')
    sys.modules['gunicorn.app.base'] = g_app_base
    g_app_base.BaseApplication = BaseApplication
    
    g_arbiter = types.ModuleType('gunicorn.arbiter')
    sys.modules['gunicorn.arbiter'] = g_arbiter
    g_arbiter.Arbiter = Arbiter

    # Other common gunicorn imports
    sys.modules['gunicorn.util'] = types.ModuleType('gunicorn.util')
    sys.modules['gunicorn.config'] = types.ModuleType('gunicorn.config')
    sys.modules['gunicorn.errors'] = types.ModuleType('gunicorn.errors')
    sys.modules['gunicorn.glogging'] = types.ModuleType('gunicorn.glogging')
    sys.modules['gunicorn.workers'] = types.ModuleType('gunicorn.workers')
