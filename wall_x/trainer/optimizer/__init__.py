from .utils import get_optimizer, register_optimizer

# DMuon is optional (external package). Skip registration if dmuon isn't
# installed so wall_x still imports; the registry will simply not have
# "dmuon" and get_optimizer("dmuon", ...) will raise a clear KeyError.
try:
    from . import dmuon  # noqa: F401 - side-effect registers "dmuon"
except ImportError:
    pass
