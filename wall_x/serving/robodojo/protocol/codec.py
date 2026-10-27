"""msgpack encode/decode for WebSocket binary frames."""

from __future__ import annotations

from typing import Any

import cv2
import msgpack
import msgpack_numpy
import numpy as np
from pydantic import ValidationError

from wall_x.serving.robodojo.protocol.exceptions import ErrorCode, WsError
from wall_x.serving.robodojo.protocol.schemas import Frame

_JPEG_KEY = b"**jpg**"
_JPEG_SHAPE_KEY = b"**shape**"


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy arrays are not supported"
            )
        return msgpack_numpy.encode(obj)
    if isinstance(obj, np.generic):
        if obj.dtype.kind == "O":
            raise WsError(
                ErrorCode.INVALID_FRAME, "object dtype numpy scalars are not supported"
            )
    return msgpack_numpy.encode(obj)


def _decode_jpeg_ndarray(obj: dict[Any, Any]) -> np.ndarray:
    jpeg_bytes = obj.get(_JPEG_KEY)
    shape = obj.get(_JPEG_SHAPE_KEY)

    if not isinstance(jpeg_bytes, (bytes, bytearray)):
        raise ValueError("JPEG ndarray payload must contain bytes in b'**jpg**'")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError("JPEG ndarray shape must be [H, W, 3]")
    if any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or int(dimension) <= 0
        for dimension in shape
    ):
        raise ValueError("JPEG ndarray shape dimensions must be positive integers")

    expected_shape = tuple(int(dimension) for dimension in shape)
    if expected_shape[2] != 3:
        raise ValueError("JPEG ndarray shape must have exactly 3 channels")

    encoded = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("failed to JPEG-decode image payload")
    if decoded.shape != expected_shape:
        raise ValueError(
            "JPEG ndarray decoded shape mismatch: "
            f"expected={expected_shape}, actual={decoded.shape}"
        )
    if decoded.dtype != np.uint8:
        raise ValueError(
            f"JPEG ndarray decoded dtype must be uint8, got {decoded.dtype}"
        )
    return np.ascontiguousarray(decoded)


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    if _JPEG_KEY in obj:
        return _decode_jpeg_ndarray(obj)
    if obj.get(b"nd") is True and obj.get(b"kind") == b"O":
        raise ValueError("object dtype numpy arrays are not supported")
    decoded = msgpack_numpy.decode(obj)
    if isinstance(decoded, np.ndarray) and decoded.dtype.kind == "O":
        raise ValueError("object dtype numpy arrays are not supported")
    return decoded


def encode_frame(frame: Frame | dict[str, Any]) -> bytes:
    if isinstance(frame, Frame):
        wire = frame.to_wire_dict()
    else:
        wire = dict(frame)
    try:
        return msgpack.packb(wire, default=_encode_numpy, use_bin_type=True)
    except WsError:
        raise
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack encode failed: {exc}") from exc


def decode_frame(data: bytes) -> dict[str, Any]:
    try:
        obj = msgpack.unpackb(data, raw=False, object_hook=_decode_numpy)
    except Exception as exc:
        raise WsError(ErrorCode.INVALID_FRAME, f"msgpack decode failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise WsError(ErrorCode.INVALID_FRAME, "frame must be a msgpack map")
    return obj


def decode_envelope(data: bytes) -> Frame:
    try:
        return Frame.from_wire_dict(decode_frame(data))
    except WsError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise WsError(
            ErrorCode.INVALID_FRAME, f"invalid frame envelope: {exc}"
        ) from exc
