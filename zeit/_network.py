"""Telling a computer without internet access (a cluster's compute node, often) from a server
that answered with an error, and saying what to do about it."""

import errno
import socket
from typing import Optional

# The connection itself failed: no name resolution, no route, refused, timed out.
_ERRNOS = {errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT, errno.ECONNRESET,
           errno.ENETDOWN, errno.EHOSTDOWN}

OFFLINE_HINT = ("This computer seems to have no internet access (common on the compute nodes of a cluster): "
                "download what is needed beforehand on a computer that has it, such as the cluster's login "
                "node")


def connection_failed(error: BaseException) -> bool:
    """Whether ``error`` (or what caused it) is a failure to reach the server at all, and not an
    answer from it (an HTTP error, a refused credential)."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        name = type(error).__name__
        if isinstance(error, (socket.gaierror, socket.timeout, TimeoutError, ConnectionRefusedError,
                              ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(error, OSError) and error.errno in _ERRNOS:
            return True
        # requests/urllib3 and httplib2 (Earth Engine) wrap them in their own classes
        if name in ("NameResolutionError", "NewConnectionError", "ConnectTimeoutError", "ConnectTimeout",
                    "ServerNotFoundError"):
            return True
        if name == "ConnectionError" and type(error).__module__.startswith("requests"):
            return True
        reason = getattr(error, "reason", None)     # urllib.error.URLError
        if isinstance(reason, BaseException):
            error = reason
            continue
        error = error.__cause__ or error.__context__
    return False


def offline_message(what: str, fetch: Optional[str] = None) -> str:
    """The message for ``what`` that could not be reached; ``fetch``: the command that
    downloads it beforehand."""
    text = f"could not reach {what}. {OFFLINE_HINT}"
    return text + (f", with: {fetch}" if fetch else "") + "."


def probe(url: str, timeout: float = 5.0) -> Optional[str]:
    """None when ``url`` answers (with any HTTP status), else why not, in a few words."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "zeit"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return None
    except urllib.error.HTTPError:
        return None
    except Exception as e:  # noqa: BLE001 - reported, not raised
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.gaierror):
            return "the name does not resolve (no DNS)"
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return f"no answer in {timeout:g} s"
        if isinstance(reason, ConnectionRefusedError):
            return "connection refused"
        return str(reason) or type(reason).__name__
