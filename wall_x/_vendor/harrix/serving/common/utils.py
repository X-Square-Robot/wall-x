"""Small shared helpers for the serving stack (leaf module — no serving imports)."""

import urllib.error
import urllib.request


def probe_http_health(url: str, timeout_s: float) -> tuple[bool, str | None]:
    """Blocking HTTP health probe → ``(healthy, error_message)``.

    ``healthy`` is True iff the endpoint returns 2xx within ``timeout_s``;
    otherwise the message describes the non-2xx status or the transport error.
    Run via ``asyncio.to_thread`` from async callers (router health loop,
    worker readiness wait).
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"health returned HTTP {response.status}"
    except urllib.error.URLError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
