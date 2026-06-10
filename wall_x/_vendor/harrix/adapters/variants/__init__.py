"""Register inference adapter variants through import side effects."""

from __future__ import annotations

import importlib
import pkgutil


for module_info in pkgutil.iter_modules(__path__):
    if module_info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{module_info.name}")
    except ImportError:
        continue
