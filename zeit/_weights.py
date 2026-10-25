"""Pretrained weights of ``zeit.ai``: downloaded once from where their authors publish them, at
a fixed revision, and checked against their SHA-256 before use. Kept apart from ``zeit.ai``
(and from torch) so that ``zeit fetch weights`` can download them on a computer that does not
run the models, such as a cluster's login node."""
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PRESTO_URL = ("https://raw.githubusercontent.com/nasaharvest/presto/"
              "11e207a668a34336ced1d8e492a1bd5849b96c4a/data/default_model.pt")
PRESTO_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"

PRITHVI_HUB = "https://huggingface.co/ibm-nasa-geospatial"
#: Prithvi-EO-2.0's published models: repository, revision, file, SHA-256; width, depth, heads,
#: patch side; whether they encode the dates and the location (TL).
PRITHVI_VARIANTS = {
    "tiny-TL": ("Prithvi-EO-2.0-tiny-TL", "335eadc2c45ad5abe7bd307223e1c48c5b60c41b", "Prithvi_EO_V2_tiny_TL.pt",
                "d47326db9bad502b611f73e3e3f3a0e68b7b82640d67c22c795417f7209f8d70", 192, 12, 3, 16, True),
    "100M-TL": ("Prithvi-EO-2.0-100M-TL", "2c84e383194986040f883cc43d7869002c425e1b", "Prithvi_EO_V2_100M_TL.pt",
                "d45406d5fc51af1d9657d48f2e2c3ff077408a2e1113f9a242889a4fe4469b17", 768, 12, 12, 16, True),
    "300M": ("Prithvi-EO-2.0-300M", "9eb1b1102806593963daa333bcc491b1c6f8562f", "Prithvi_EO_V2_300M.pt",
             "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7", 1024, 24, 16, 16, False),
    "300M-TL": ("Prithvi-EO-2.0-300M-TL", "63adbd39c271da4c42f447e69b1a7c91a338cdc9", "Prithvi_EO_V2_300M_TL.pt",
                "3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb", 1024, 24, 16, 16, True),
    "600M": ("Prithvi-EO-2.0-600M", "2a5cf6cedea1c0b31734021e2a5ac90082825414", "Prithvi_EO_V2_600M.pt",
             "26a3f387ac6fc8e5eee8eb1986e6b9e99f36aa4d03aeb7fd5722feee5d5f88d0", 1280, 32, 16, 14, False),
    "600M-TL": ("Prithvi-EO-2.0-600M-TL", "3d72adc6dfc4cc3862bf4f41da8c83db267193c9", "Prithvi_EO_V2_600M_TL.pt",
                "7b92c53b0204a76bb775bd8930f045e05776251caa8c83f7367ed0b75b594702", 1280, 32, 16, 14, True),
}


def published() -> Dict[str, Tuple[str, str, str]]:
    """Every published weight file, by the name ``zeit fetch weights`` takes: (URL, SHA-256,
    what it is)."""
    found = {"presto": (PRESTO_URL, PRESTO_SHA256, "Presto's weights")}
    for variant, (repo, rev, name, sha) in ((v, spec[:4]) for v, spec in PRITHVI_VARIANTS.items()):
        found[f"prithvi-{variant.lower()}"] = (f"{PRITHVI_HUB}/{repo}/resolve/{rev}/{name}", sha,
                                                f"the weights of {repo}")
    return found


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _name_of(url: str) -> str:
    """The name ``zeit fetch weights`` takes for the file at ``url``."""
    return next((name for name, (u, _, _) in published().items() if u == url), "all")


def fetch(url: str, sha256: str, what: str) -> Path:
    """The file at ``url``: from the folder ``ZEIT_WEIGHTS`` names when it is there (a copy
    made elsewhere, for a computer without network access), else from the cache (the
    ``weights`` folder of ``ZEIT_CACHE_DIR``, by default ``~/.cache/zeit``), downloaded on first
    use. Checked once against ``sha256`` (a ``.sha256`` file beside it records that)."""
    from ._cache import cache_dir
    from ._network import connection_failed, offline_message

    name = url.rsplit("/", 1)[-1]
    given = os.environ.get("ZEIT_WEIGHTS")
    if given and (Path(given) / name).exists():
        path = Path(given) / name
    else:
        path = cache_dir("weights") / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(name + ".part")
            print(f"zeit: downloading {what} to {path}", file=sys.stderr)
            try:
                with urllib.request.urlopen(url) as response, open(part, "wb") as out:
                    shutil.copyfileobj(response, out, 1 << 22)
            except OSError as e:
                part.unlink(missing_ok=True)
                if connection_failed(e):
                    raise RuntimeError(offline_message(f"{what} ({url})", f"zeit fetch weights {_name_of(url)}")
                                       + f" (or copy {name} into a folder and set ZEIT_WEIGHTS to it)") from e
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


def fetch_weights(*names: str) -> List[Path]:
    """Download the weights ``names`` (``"presto"``, ``"prithvi-300m"``, ...; ``"all"``, or
    none, for every one) into the cache, for a computer that will use them without network
    access (a cluster's compute nodes, when they see the cache: set ``ZEIT_CACHE_DIR`` to a
    shared folder). The same as ``zeit fetch weights``."""
    available = published()
    wanted = [n.lower() for n in names] or ["all"]
    if "all" in wanted:
        wanted = list(available)
    unknown = [n for n in wanted if n not in available]
    if unknown:
        raise ValueError(f"no weights {', '.join(unknown)}; the published ones: {', '.join(available)}")
    return [fetch(*available[n]) for n in dict.fromkeys(wanted)]
