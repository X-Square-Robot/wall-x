from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_FRAME = "invalid_frame"
    UNKNOWN_MESSAGE_TYPE = "unknown_message_type"
    TIMEOUT = "timeout"
    INFER_FAILED = "infer_failed"
    RESET_FAILED = "reset_failed"
    INTERNAL = "internal"


class WsError(Exception):
    def __init__(
        self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
