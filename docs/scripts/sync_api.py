"""Keep the API reference in sync with the code.

Every API entry in ``docs/api/*.md`` starts with a marker naming the object:

    <!-- sig: zeit.landtrendr -->
    ```python
    ...generated...
    ```

Running this script (from the repository root, with Zeit importable):

    python docs/scripts/sync_api.py           # rewrite the signature blocks
    python docs/scripts/sync_api.py --check   # only report problems, exit 1 if any

regenerates each signature block from ``inspect.signature`` and checks the
parameter table that follows it (inside ``<div class="params" markdown>``):
every documented parameter must exist, every parameter should be documented,
and documented defaults must match the code.
"""

import importlib
import inspect
import re
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1] / "api"
MARKER = re.compile(r"<!-- sig: ([\w.]+) -->\n```python\n.*?```", re.S)
WIDTH = 70


def resolve(path):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(path)


def fmt_default(value):
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    return repr(value)


def display_name(path):
    # Methods of the xarray accessor read as `DataArray.zeit.method`.
    if ".ZeitAccessor." in path:
        return "DataArray.zeit." + path.rsplit(".", 1)[1]
    if ".ZeitDatasetAccessor." in path:
        return "Dataset.zeit." + path.rsplit(".", 1)[1]
    return path


def parameters(obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    params = list(inspect.signature(target).parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    return params


def render(path):
    obj = resolve(path)
    parts = []
    for p in parameters(obj):
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={fmt_default(p.default)}")
    prefix = ("class " if inspect.isclass(obj) else "") + display_name(path)
    one_line = f"{prefix}({', '.join(parts)})"
    if len(one_line) <= WIDTH:
        return one_line
    # Pack parameters onto indented lines of at most WIDTH characters.
    lines, line = [], "   "
    for a in parts:
        if len(line) + len(a) + 2 > WIDTH and line.strip():
            lines.append(line)
            line = "   "
        line += f" {a},"
    lines.append(line)
    return prefix + "(\n" + "\n".join(lines) + "\n)"


def documented(section):
    """(name, default) pairs from the first params table of an API section."""
    m = re.search(r'<div class="params" markdown>\s*\n(.*?)</div>', section, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].startswith("`"):
            continue
        names = [n.lstrip("*") for n in re.findall(r"`([\w*]+)`", cells[0])]
        defaults = re.findall(r"`([^`]*)`", cells[2])
        if len(names) > 1 and len(defaults) == len(names):
            rows.extend(zip(names, defaults))
        else:
            rows.extend((n, cells[2] if len(names) == 1 else "") for n in names)
    return rows


def norm_default(text):
    text = text.strip().strip("`").replace("'", '"')
    return {"required": None, "**required**": None, "—": None, "": None}.get(text.lower(), text)


def main(check_only):
    problems = []
    for md in sorted(API_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")

        def sub(m):
            path = m.group(1)
            try:
                return f"<!-- sig: {path} -->\n```python\n{render(path)}\n```"
            except Exception as e:  # noqa: BLE001
                problems.append(f"{md.name}: cannot resolve {path}: {e}")
                return m.group(0)

        new = MARKER.sub(sub, text)

        # Check each section's parameter table against the signature.
        chunks = re.split(r"(?=<!-- sig: )", new)
        for chunk in chunks[1:]:
            path = re.match(r"<!-- sig: ([\w.]+) -->", chunk).group(1)
            section = re.split(r"\n#{2,3} ", chunk)[0]
            rows = documented(section)
            if rows is None:
                continue
            try:
                params = {p.name: p for p in parameters(resolve(path))}
            except Exception:  # noqa: BLE001
                continue
            has_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
            names = {n for n, _ in rows}
            for name, default in rows:
                if name not in params and not has_kwargs:
                    problems.append(f"{md.name}: {path}: documents unknown parameter `{name}`")
                elif name in params:
                    p = params[name]
                    want = None if p.default is p.empty else fmt_default(p.default).replace("'", '"')
                    got = norm_default(default)
                    if got is not None and want is not None and got != want:
                        problems.append(f"{md.name}: {path}: `{name}` default is {want}, docs say {got}")
            for name, p in params.items():
                if name not in names and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                    problems.append(f"{md.name}: {path}: parameter `{name}` is not documented")

        if new != text and not check_only:
            md.write_text(new, encoding="utf-8")
            print(f"updated {md.relative_to(API_DIR.parents[1])}")

    for p in problems:
        print("  " + p)
    print(f"{len(problems)} problem(s)")
    return 1 if (problems and check_only) else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
