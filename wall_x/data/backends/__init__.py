"""Data backend registration.

Importing this package registers the default shipped backends. Additional
backends are loaded by plugin modules so their names and dependencies do not
have to appear in the default import path.
"""

import importlib

from wall_x.data._registry import record_import_error

for _name in ("lerobot",):
    try:
        importlib.import_module(f"wall_x.data.backends.{_name}")
    except ImportError as _e:
        record_import_error(_name, _e)
