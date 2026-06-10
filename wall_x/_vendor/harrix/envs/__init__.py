"""Environment registrations for harrix."""

_LIBERO_IMPORT_ERROR = None

try:
    from wall_x._vendor.harrix.envs import libero  # noqa: F401
except ModuleNotFoundError as exc:
    _LIBERO_IMPORT_ERROR = exc
