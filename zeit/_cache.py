"""Where zeit keeps what it downloads: one root for every cache.

The root is ``ZEIT_CACHE_DIR`` when it is set (on a cluster, a folder in the scratch space:
the home directory's quota is small, and some compute nodes see it read-only), else
``$XDG_CACHE_HOME/zeit``, else ``%LOCALAPPDATA%\\zeit`` on Windows and ``~/.cache/zeit``
elsewhere. In it, one folder per cache:

- ``weights``: the pretrained weights of ``zeit.ai`` (``ZEIT_WEIGHTS`` names a folder of copies
  read before it);
- ``embeddings``: the downloads of ``zeit.load_embeddings`` (``cache_dir=`` names another);
- ``tiles``: the tiles a tile run built or received (``zeit worker --cache-dir`` names another);
- ``map-tiles``: the basemap tiles of ``zeit.plot``.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

CACHES = ("weights", "embeddings", "tiles", "map-tiles")


def cache_root() -> Path:
    """The folder of every cache (see the module)."""
    given = os.environ.get("ZEIT_CACHE_DIR")
    if given:
        return Path(os.path.expanduser(given))
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "zeit"
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "zeit"
    return Path.home() / ".cache" / "zeit"


def _legacy(name: str) -> Optional[Path]:
    """Where zeit 2.2 and before kept a cache, when it differs from today's."""
    if name in ("weights", "embeddings"):
        return Path.home() / ".cache" / "zeit" / name
    if name == "tiles" and sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "zeit" / "tiles-cache"
    return None


def cache_dir(name: str) -> Path:
    """The folder of the cache ``name`` (one of ``CACHES``), not created. A cache that an
    older zeit filled elsewhere is still used there, unless ``ZEIT_CACHE_DIR`` is set."""
    if name not in CACHES:
        raise ValueError(f"no cache {name!r}; one of {', '.join(CACHES)}")
    path = cache_root() / name
    if not os.environ.get("ZEIT_CACHE_DIR") and not path.exists():
        old = _legacy(name)
        if old is not None and old != path and old.is_dir():
            return old
    return path


def writable(path: Path) -> bool:
    """Whether a file can be written in ``path`` (or in the nearest folder above it that
    exists): tried, since permissions alone mislead on network file systems and Windows."""
    import tempfile

    probe = Path(path)
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    try:
        with tempfile.TemporaryFile(dir=probe):
            return True
    except OSError:
        return False


def caches() -> Dict[str, Path]:
    """Every cache's folder."""
    return {name: cache_dir(name) for name in CACHES}
