"""The notebook viewer: an anywidget carrying the browser's requests to a Session."""

import pathlib
import traceback

from ._session import Session

HERE = pathlib.Path(__file__).parent


def in_notebook() -> bool:
    """True inside a Jupyter kernel (JupyterLab, Notebook, VS Code, Colab)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ in ("ZMQInteractiveShell", "Shell")


_VIEWER_CLASS = None


def viewer_class():
    """The anywidget class, built once (anywidget is imported only when a viewer is made)."""
    global _VIEWER_CLASS
    if _VIEWER_CLASS is not None:
        return _VIEWER_CLASS
    try:
        import anywidget
        import traitlets
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("the interactive viewer needs anywidget: pip install anywidget "
                          "(or zeit-cdts[plot]); use static=True for a matplotlib figure") from err

    class Viewer(anywidget.AnyWidget):
        """Interactive map viewer (zeit.plot)."""

        _esm = HERE / "viewer.js"
        _css_text = traitlets.Unicode((HERE / "viewer.css").read_text(encoding="utf-8")).tag(sync=True)
        height = traitlets.Int(480).tag(sync=True)
        fps = traitlets.Int(8).tag(sync=True)

        def __init__(self, session: Session, **kwargs):
            super().__init__(**kwargs)
            self.session = session
            self.on_msg(self._on_request)

        def _on_request(self, _widget, content, buffers):
            if content.get("type") != "request":
                return
            reply, out = handle_safely(self.session, content["request"])
            self.send({"type": "reply", "id": content["id"], **reply}, buffers=out)

    _VIEWER_CLASS = Viewer
    return Viewer


def handle_safely(session: Session, request):
    """({"content": ...} or {"error": ...}, buffers): errors go to the browser, not the kernel."""
    try:
        content, buffers = session.handle(request)
        return {"content": content}, buffers
    except Exception as err:  # noqa: BLE001
        return {"error": f"{type(err).__name__}: {err}", "trace": traceback.format_exc()[-2000:]}, []


def make_widget(session: Session, *, height: int = 480, fps: int = 8):
    return viewer_class()(session, height=height, fps=fps)


def interpret_bundle() -> str:
    """viewer.js and interpret.js as one module: anywidget loads a single module, and the
    window serves the same text. The viewer's own default export (its widget) gives way to
    the interpreter's."""
    viewer = (HERE / "viewer.js").read_text(encoding="utf-8")
    marker = "export default {"
    if viewer.count(marker) != 1:
        raise RuntimeError("viewer.js: expected one default export")
    return viewer.replace(marker, "const viewerWidget = {") + chr(10) + (HERE / "interpret.js").read_text(encoding="utf-8")


_INTERPRET_CLASS = None


def make_interpret_widget(session: Session, *, height: int = 480):
    """The notebook widget of zeit.interpret."""
    global _INTERPRET_CLASS
    if _INTERPRET_CLASS is None:
        import traitlets

        base = viewer_class()

        class InterpretViewer(base):
            """Reference labelling (zeit.interpret)."""

            _esm = interpret_bundle()
            _css_text = traitlets.Unicode((HERE / "viewer.css").read_text(encoding="utf-8") + chr(10)
                                          + (HERE / "interpret.css").read_text(encoding="utf-8")).tag(sync=True)

        _INTERPRET_CLASS = InterpretViewer
    return _INTERPRET_CLASS(session, height=height)
