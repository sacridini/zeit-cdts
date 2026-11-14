"""Shared helpers of the field-boundary validation download scripts."""

import datetime
import json
import os
import time
from pathlib import Path

import requests

USER_AGENT = "zeit-validation-script"
DEFAULT_DIR = r"C:\Users\eduar\Documents\dev\zeit_data\fields"

# Municipalities of the test areas: name -> IBGE code.
MUNICIPALITIES = {
    "porto_murtinho": "5006903",
    "sorriso": "5107925",
    "lem": "2919553",  # Luis Eduardo Magalhaes (BA), checked against IBGE
}


def data_dir():
    """Return the data folder (ZEIT_FIELDS_DATA or the default), created."""
    p = Path(os.environ.get("ZEIT_FIELDS_DATA", DEFAULT_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Counter:
    """Bytes transferred by this process."""

    bytes = 0


def session():
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get(url, sess=None, params=None, tries=5, timeout=120):
    """GET with retries and exponential backoff; counts the bytes."""
    sess = sess or session()
    for k in range(tries):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            Counter.bytes += len(r.content)
            return r
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            wait = 2 ** (k + 1)
            print(f"  retry in {wait}s after {e}")
            time.sleep(wait)


def today():
    return datetime.date.today().isoformat()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def done(name):
    print(f"[{name}] bytes transferred: {Counter.bytes:,}")
