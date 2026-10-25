import json
import os
import sys
from typing import Optional

import ee

HIGH_VOLUME = 'https://earthengine-highvolume.googleapis.com'


def initialize_gee(project: Optional[str] = None, *, key_file: Optional[str] = None) -> None:
    """
    Initializes the Google Earth Engine Python API.

    With a service account's JSON key (``key_file``, or the file ``GOOGLE_APPLICATION_CREDENTIALS``
    names), it signs in as that account: what a batch job on a cluster can do without a
    browser. Otherwise it uses the credentials ``earthengine authenticate`` saved, and when
    there are none asks to authenticate, if there is a terminal to ask in; in a batch job it
    raises a RuntimeError that says how to authenticate instead of waiting for an answer.

    Args:
        project (str, optional): The Google Cloud Project ID to use (default: the service
            account's project, or the one saved with the credentials).
        key_file (str, optional): A service account's JSON key.
    """
    key_file = key_file or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    try:
        if key_file:
            with open(key_file) as f:
                key = json.load(f)
            credentials = ee.ServiceAccountCredentials(key.get('client_email'), key_file)
            ee.Initialize(credentials, project=project or key.get('project_id'), opt_url=HIGH_VOLUME)
            return
        _initialize(project)
    except Exception as e:
        _explain(e, key_file)
        if not _can_ask():
            raise RuntimeError(
                f"Earth Engine could not start ({e}), and this process has no terminal to authenticate in "
                "(a batch job?). Authenticate once where there is a browser, with `earthengine authenticate` "
                "(on a cluster, on the login node: the credentials go to ~/.config/earthengine, which the "
                "compute nodes usually see), or use a service account: set GOOGLE_APPLICATION_CREDENTIALS to "
                "its JSON key.") from e
        print("GEE authentication required. Please authenticate...")
        ee.Authenticate()
        _initialize(project)


def _can_ask() -> bool:
    """A terminal, or a notebook (where ee.Authenticate shows a link): somewhere to answer."""
    if sys.stdin is not None and sys.stdin.isatty():
        return True
    shell = sys.modules.get("IPython")
    try:
        return shell is not None and shell.get_ipython() is not None
    except Exception:  # noqa: BLE001
        return False


def _initialize(project: Optional[str]) -> None:
    if project:
        ee.Initialize(project=project, opt_url=HIGH_VOLUME)
    else:
        ee.Initialize(opt_url=HIGH_VOLUME)


def _explain(error: Exception, key_file: Optional[str]) -> None:
    """Raise what authenticating again would not fix: no network, or a service account's key
    that does not work."""
    from .._network import connection_failed, offline_message

    if connection_failed(error):
        raise RuntimeError(offline_message(
            "Earth Engine", "zeit fetch tiles (for tiles built from Earth Engine)")) from error
    if key_file:
        raise RuntimeError(f"Earth Engine refused the service account of {key_file}: {error}") from error
