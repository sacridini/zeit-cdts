"""Pretrained weights: downloaded once from where their authors publish them, at a fixed
revision, and checked against their SHA-256 before use."""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, sha256: str, what: str) -> Path:
    """The file at ``url``: from the folder ``ZEIT_WEIGHTS`` names when it is there (a copy
    made elsewhere, for a computer without network access), else from the cache
    (``~/.cache/zeit/weights``), downloaded on first use. Checked once against ``sha256``
    (a ``.sha256`` file beside it records that)."""
    name = url.rsplit("/", 1)[-1]
    given = os.environ.get("ZEIT_WEIGHTS")
    if given and (Path(given) / name).exists():
        path = Path(given) / name
    else:
        path = Path.home() / ".cache" / "zeit" / "weights" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(name + ".part")
            print(f"zeit: downloading {what} to {path}", file=sys.stderr)
            try:
                with urllib.request.urlopen(url) as response, open(part, "wb") as out:
                    shutil.copyfileobj(response, out, 1 << 22)
            except OSError as e:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"could not download {what} from {url}: {e}. Without network access, "
                                   f"download {name} elsewhere and set ZEIT_WEIGHTS to the folder holding it") from e
            part.replace(path)
    stamp = path.with_name(path.name + ".sha256")
    if not (stamp.exists() and stamp.read_text().strip() == sha256):
        digest = _sha256(path)
        if digest != sha256:
            raise RuntimeError(f"{path} is not {what} (SHA-256 {digest}, expected {sha256}); delete it to "
                               "download it again")
        try:
            stamp.write_text(sha256)
        except OSError:
            pass   # a read-only copy: checked again next time
    return path
