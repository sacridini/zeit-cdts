"""``zeit config``: the settings zeit keeps for this user, in ``config.toml`` of the config
folder (``ZEIT_CONFIG_DIR``, or ``%APPDATA%\\zeit`` on Windows, ``~/.config/zeit`` elsewhere),
for those who would rather not depend on an environment variable (on Windows, a variable
for the whole system needs an administrator).

``threads``: the threads of ``n_jobs=-1`` outside a batch job, read by the core as well
(``rust/zeit-core/src/parallel.rs``), after ``ZEIT_NUM_THREADS`` and before
``OMP_NUM_THREADS``. ``auto`` removes it.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# The settings and what each takes.
SETTINGS = {"threads": "a number of threads, or auto (zeit decides; the default)"}


def config_dir() -> Path:
    env = os.environ.get("ZEIT_CONFIG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home()) / "zeit"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "zeit"


def config_file() -> Path:
    return config_dir() / "config.toml"


def read() -> Dict[str, str]:
    """The settings of config.toml (its ``key = value`` lines)."""
    try:
        text = config_file().read_text(encoding="utf-8")
    except OSError:
        return {}
    found = {}
    for line in text.splitlines():
        key, eq, value = line.split("#", 1)[0].partition("=")
        if eq and key.strip():
            found.setdefault(key.strip(), value.strip().strip('"'))
    return found


def threads() -> Optional[int]:
    """``zeit config threads``, or None (auto)."""
    try:
        n = int(read().get("threads", ""))
    except ValueError:
        return None
    return n if n > 0 else None


def set_setting(key: str, value: Optional[str]) -> Path:
    """Writes ``key = value`` in config.toml (None removes it), keeping the other lines."""
    if key not in SETTINGS:
        raise ValueError(f"no setting {key!r}: {', '.join(SETTINGS)}")
    path = config_file()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    lines = [line for line in lines if line.split("#", 1)[0].partition("=")[0].strip() != key]
    if value is not None:
        lines.append(f"{key} = {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    try:
        from . import _core

        _core.parallel.refresh()     # this process sees it at once, not in 30 seconds
    except (ImportError, AttributeError):
        pass
    return path


def run(key: Optional[str], value: Optional[str]) -> None:
    """``zeit config [KEY [VALUE]]``."""
    if key is None:
        found = read()
        print(f"{config_file()}{'' if found else ' (no settings)'}")
        for name, about in SETTINGS.items():
            print(f"  {name} = {found.get(name, 'auto')}    ({about})")
        return
    if key not in SETTINGS:
        raise SystemExit(f"zeit config: no setting {key!r} (settings: {', '.join(SETTINGS)})")
    if value is None:
        print(read().get(key, "auto"))
        return
    if value == "auto":
        path = set_setting(key, None)
        print(f"threads: auto (removed from {path})")
        return
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n <= 0:
        raise SystemExit(f"zeit config threads: {value!r} is not a number of threads above 0, or auto")
    path = set_setting(key, str(n))
    print(f"threads = {n}, in {path}: the threads of n_jobs=-1 outside a batch job, unless "
          "ZEIT_NUM_THREADS is set")
