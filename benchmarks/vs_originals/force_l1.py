"""The Sentinel-2 Level 1C steps of zeit-core (``zeit_core::s2l1c``, a port of FORCE 3.11.00's ``parse_metadata_sentinel2``
of ``meta-ll.c`` and of ``bounds_level1`` and ``convert_level1`` of ``read-ll.c``) against FORCE's own functions, on the
metadata files of real granules.

The C side is ``force_l1/`` (``make_harness.sh``, WSL): a ``main.c`` compiled against FORCE's headers and linked with the
object files FORCE's ``make`` built for ``force-l2ps``, so that ``parse_metadata`` (the real one: it reads the product file
``MTD_MSIL1C.xml`` and the granule file ``MTD_TL.xml`` of a .SAFE directory this script lays out, finds the 13 band files
``*_Bxx.jp2``, which are 1x1 GeoTIFFs here, and takes the projection of the first), ``bounds_level1`` and ``convert_level1``
run as the L2PS runs them. ``read_level1`` (the reading of the 13 rasters) is not run: the digital numbers are given, in
a brick of the size of the given image. Zeit's side is ``zeit._core.s2l1c.run`` (the clone's, checked below).

Granules (the XML files of the Earth Search ``sentinel-2-l1c`` items, fetched from the public ``sentinel-s2-l1c`` bucket
into the work directory, not into the repository): ``S2B_23LKC_20210728_0_L1C`` (processing baseline 03.01, no offsets) and
``S2B_23LKC_20230728_0_L1C`` (05.09, ``RADIO_ADD_OFFSET`` -1000 in every band); and the partial tile of the second
23LKC date of ``tests/data`` (a granule file with cells without data, with the product file of the 05.09 granule). Variants
of the product file: Sentinel-2A and 2C, a quantification value of 1000. Each granule runs with and without the clipping of
the image to the valid cells (``doreproj``), and the pixels with and without the buffer of the no-data (``BUFFER_NODATA``).
The digital numbers are synthetic (random in the range of real data, with no-data blocks, saturated blobs, values below
1000 and a cirrus band that is 0 in places).

    python benchmarks/vs_originals/force_l1.py [--skip-build]

Prints, per scene and field, the number of values compared and how many are equal bit for bit; exits with 1 if anything
differs.
"""
import argparse
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))  # the clone's zeit, not an installed one

from common import no_power_throttling, workdir  # noqa: E402
from force_atmo import _wsl  # noqa: E402

DATA = workdir("force_l1")
EXE = "$HOME/force-ref-work/l1/l1_harness"
NB = 13
BAND_FILES = ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]
S3 = "https://sentinel-s2-l1c.s3.amazonaws.com"
GRANULES = {
    # name: (tile path, product path, name of the granule directory, EPSG)
    "baseline 03.01 (2021-07-28)": (
        "tiles/23/L/KC/2021/7/28/0/metadata.xml",
        "products/2021/7/28/S2B_MSIL1C_20210728T132239_N0301_R038_T23LKC_20210728T151918/metadata.xml",
        "S2B_OPER_MSI_L1C_TL_VGS2_20210728T151918_A022944_T23LKC_N03.01",
        32723,
    ),
    "baseline 05.09 (2023-07-28)": (
        "tiles/23/L/KC/2023/7/28/0/metadata.xml",
        "products/2023/7/28/S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149/metadata.xml",
        "L1C_T23LKC_A033383_20230728T132239",
        32723,
    ),
}
PARTIAL_TL = REPO / "tests" / "data" / "s2_23LKC_20230726_partial_MTD_TL.xml"


def fetch(url, dest):
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest.read_bytes()


def tiny_geotiff(epsg):
    """A 1x1 uint16 GeoTIFF in EPSG:`epsg` (classic TIFF, little endian, one strip)."""
    entries = []  # (tag, type, count, value-or-offset-bytes)
    extra = bytearray()
    ifd_n = 14
    ifd_size = 2 + 12 * ifd_n + 4
    base = 8 + ifd_size
    data_off = base
    extra += struct.pack("<H", 0)  # the pixel
    scale_off = base + len(extra)
    extra += struct.pack("<3d", 10.0, 10.0, 0.0)
    tie_off = base + len(extra)
    extra += struct.pack("<6d", 0.0, 0.0, 0.0, 300000.0, 7000000.0, 0.0)
    geo_off = base + len(extra)
    extra += struct.pack("<16H", 1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg)
    short = lambda v: struct.pack("<HH", v, 0)  # noqa: E731
    entries = [
        (256, 3, 1, short(1)), (257, 3, 1, short(1)), (258, 3, 1, short(16)), (259, 3, 1, short(1)),
        (262, 3, 1, short(1)), (273, 4, 1, struct.pack("<I", data_off)), (277, 3, 1, short(1)),
        (278, 3, 1, short(1)), (279, 4, 1, struct.pack("<I", 2)), (284, 3, 1, short(1)),
        (33550, 12, 3, struct.pack("<I", scale_off)), (33922, 12, 6, struct.pack("<I", tie_off)),
        (34735, 3, 16, struct.pack("<I", geo_off)), (339, 3, 1, short(1)),
    ]
    entries.sort()
    assert len(entries) == ifd_n
    out = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", ifd_n)
    for tag, typ, cnt, val in entries:
        out += struct.pack("<HHI", tag, typ, cnt) + val
    out += struct.pack("<I", 0) + bytes(extra)
    return out


def lay_out(name, product, tile, granule, epsg):
    """A .SAFE directory (product file, granule file, 13 band files); returns the granule directory."""
    safe = DATA / name / "X.SAFE"
    gran = safe / "GRANULE" / granule
    img = gran / "IMG_DATA"
    img.mkdir(parents=True, exist_ok=True)
    (safe / "MTD_MSIL1C.xml").write_bytes(product)
    (gran / "MTD_TL.xml").write_bytes(tile)
    tif = tiny_geotiff(epsg)
    for b in BAND_FILES:
        (img / f"T23LKC_20230728T132239_B{b}.jp2").write_bytes(tif)
    return gran


def synthetic_dn(rng, nx, ny):
    """13 bands of digital numbers: the range of real data, with every case of bounds_level1 and convert_level1."""
    base = rng.integers(1, 12000, size=(NB, ny, nx))
    base[:, rng.random((ny, nx)) < 0.02] = rng.integers(1, 30000)  # clutter
    dn = base.astype(np.uint16)
    dn[:, : ny // 8, :] = rng.integers(1, 800, size=(NB, ny // 8, nx)).astype(np.uint16)  # below 1000: negative TOA after -1000
    dn[:, ny - ny // 6 :, : nx // 5] = 0  # a no-data block (every band), touching the border
    dn[:, ny // 3 : ny // 3 + 40, nx // 2 : nx // 2 + 60] = 0  # a no-data block inside
    dn[3, 100:120, 100:130] = 0  # one band only
    dn[10, 200:260, 300:330] = 0  # the cirrus band alone: not no-data
    dn[:, 20, 50] = 0  # single pixels, inside and on the border
    dn[:, 0, 7] = 0
    dn[:, ny - 1, nx - 1] = 0
    dn[2, ny // 2 : ny // 2 + 30, 20:50] = 65535  # saturated, one band
    dn[:, ny // 2 + 60 : ny // 2 + 70, 100:140] = 65535  # saturated, all bands
    dn[8, ny // 2 + 60, 100] = 0  # a 0 after the saturated band 2: the saturation is found first
    dn[1, ny // 2 + 80, 100] = 0
    dn[4, ny // 2 + 80, 100] = 65535  # a 0 before the saturated: off only
    dn[:, ny // 2 + 90, 100] = 65534
    dn[:, ny // 2 + 91, 100] = 1
    dn[0, ny // 2 + 92, 100] = 32768  # beyond the short after the scaling (no offset): wraps as in the C
    return np.ascontiguousarray(dn)


# the fields of the dump, in order
def parse(raw, with_pixels):
    o = {}
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, raw, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    def arr(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=pos).copy()
        pos += a.nbytes
        return a

    (o["status"],) = take("i")
    if o["status"] != 0:
        return o
    (nb,) = take("i")
    rsr, radd, scale = [], [], []
    for _ in range(nb):
        r, a, s = take("iff")
        rsr.append(r), radd.append(a), scale.append(s)
    o["rsr_band"], o["radd"], o["scale"] = np.array(rsr, np.int32), np.array(radd, np.float32), np.array(scale, np.float32)
    o["sensor"] = take("8s")[0].split(b"\0")[0].decode()
    o["dtype"], o["sat"], o["tier"] = take("iii")
    o["refsys_id"] = take("8s")[0].split(b"\0")[0].decode()
    o["date"] = np.array(take("8i"), np.int32)
    o["res"], o["ulx"], o["uly"] = take("ddd")
    o["ncols"], o["nrows"] = take("ii")
    sx, sy, o["left"], o["right"], o["top"], o["bottom"] = take("6i")
    o["s2_nx_ny"] = np.array([sx, sy], np.int32)
    (o["nodata"],) = take("f")
    for k in ("vzen", "vazi", "szen", "sazi"):
        o[k] = arr(np.float32, sx * sy)
    wl, dom = [], []
    for _ in range(nb):
        w, d = take("f16s")
        wl.append(w), dom.append(d.split(b"\0")[0].decode())
    o["wavelength"], o["domain"] = np.array(wl, np.float32), dom
    if with_pixels:
        nx, ny = take("ii")
        o["qai"] = arr(np.int16, nx * ny)
        o["toa"] = arr(np.int16, nb * nx * ny).reshape(nb, nx * ny)
    assert pos == len(raw), (pos, len(raw))
    return o


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_l1' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def run_force(name, gran, granule, subset, buf, dn):
    fout = DATA / f"{name}.out.bin"
    args = f"{_wsl(gran)} {granule} {int(subset)} 0 {int(buf)} {_wsl(fout)}"
    if dn is not None:
        fin = DATA / f"{name}.dn.bin"
        fin.write_bytes(struct.pack("<ii", dn.shape[2], dn.shape[1]) + dn.tobytes())
        args += f" {_wsl(fin)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"source ~/force-ref-work/env.sh; OMP_NUM_THREADS=1 GDAL_NUM_THREADS=1 {EXE} {args}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stdout}{r.stderr}")
    return parse(fout.read_bytes(), dn is not None)


def run_zeit(product, tile, granule, subset, buf, dn):
    from zeit._core import s2l1c

    try:
        if dn is None:
            raw = s2l1c.run(product, tile, granule, subset, buf)
        else:
            raw = s2l1c.run(product, tile, granule, subset, buf, dn.reshape(-1), dn.shape[2], dn.shape[1], 1)
    except ValueError as e:
        return {"status": 1, "error": str(e)}
    return parse(bytes(raw), dn is not None)


class Table:
    def __init__(self):
        self.bad = False
        self.tv = self.te = 0

    def row(self, title, field, a, b, note=""):
        if isinstance(a, np.ndarray) and a.dtype == np.float32:
            a, b = np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32)
        elif isinstance(a, float):
            a, b = struct.pack("<d", a), struct.pack("<d", b)
            a, b = np.array([a == b]), np.array([True])
        a, b = np.atleast_1d(np.asarray(a)), np.atleast_1d(np.asarray(b))
        if a.shape != b.shape:
            n, e = max(a.size, b.size), 0
        else:
            n, e = a.size, int((a == b).sum())
        self.tv += n
        self.te += e
        if e != n:
            self.bad = True
        print(f"{title:50s} {field:14s} {n:10d} {e:10d} {'' if e == n else '  <-- DIFFERENT'}{note}")


def compare(table, title, c, z):
    if c["status"] != z["status"]:
        table.bad = True
        print(f"{title:50s} STATUS FORCE {c['status']} Zeit {z['status']} {z.get('error', '')}")
        return
    if c["status"] != 0:
        print(f"{title:50s} both refuse ({z.get('error', '')})")
        return
    T = lambda f, a, b, note="": table.row(title, f, a, b, note)  # noqa: E731
    T("rsr, radd, scale", np.concatenate([c["rsr_band"].astype(np.float64), c["radd"], c["scale"]]),
      np.concatenate([z["rsr_band"].astype(np.float64), z["radd"], z["scale"]]), f" ({c['sensor']}, scale {c['scale'][0]:.0f}, radd {c['radd'][0]:.0f})")
    T("sensor, refsys", np.array([c["sensor"], c["refsys_id"], c["dtype"], c["sat"], c["tier"]]),
      np.array([z["sensor"], z["refsys_id"], z["dtype"], z["sat"], z["tier"]]), f" ({c['refsys_id']})")
    T("date", c["date"], z["date"], f" ({c['date'][:3].tolist()} {c['date'][4:7].tolist()})")
    T("res, ulx, uly", np.array([c["res"], c["ulx"], c["uly"]]).view(np.uint64), np.array([z["res"], z["ulx"], z["uly"]]).view(np.uint64))
    T("size, window", np.array([c[k] for k in ("ncols", "nrows", "left", "right", "top", "bottom")]),
      np.array([z[k] for k in ("ncols", "nrows", "left", "right", "top", "bottom")]),
      f" (image {c['ncols']}x{c['nrows']}, window {c['left']},{c['top']} to {c['right']},{c['bottom']})")
    T("grid size", c["s2_nx_ny"], z["s2_nx_ny"], f" ({c['s2_nx_ny'][0]}x{c['s2_nx_ny'][1]} cells)")
    T("nodata", np.array([c["nodata"]], np.float32), np.array([z["nodata"]], np.float32))
    for k in ("vzen", "vazi", "szen", "sazi"):
        T(k, c[k], z[k], f" (nodata cells {int((c[k] == c['nodata']).sum())})" if k == "vzen" else "")
    T("wavelength", c["wavelength"], z["wavelength"])
    T("domain", np.array(c["domain"]), np.array(z["domain"]))
    if "qai" in c:
        q = c["qai"]
        T("QAI", q, z["qai"], f" (off {int((q & 1).sum())}, saturated {int(((q >> 9) & 1).sum())})")
        T("TOA", c["toa"], z["toa"], f" ({int((c['toa'] == -9999).sum())} no-data, {int((c['toa'] < 0).sum() - (c['toa'] == -9999).sum())} negative)")


def blank_corner(tile, rows, cols):
    """The granule file with the first `rows` rows and `cols` columns of every angle grid set to NaN (the valid cells start
    inside the tile, so the clipped image has a window that does not start at 0)."""
    out, k = [], -1
    for line in tile.split(b"\n"):
        if b"<Values_List>" in line:
            k = 0
        elif b"</Values_List>" in line:
            k = -1
        elif k >= 0 and b"<VALUES>" in line:
            head, rest = line.split(b"<VALUES>")
            vals, tail = rest.split(b"</VALUES>")
            v = vals.split(b" ")
            if k < rows:
                v = [b"NaN"] * len(v)
            else:
                v[:cols] = [b"NaN"] * min(cols, len(v))
            line = head + b"<VALUES>" + b" ".join(v) + b"</VALUES>" + tail
            k += 1
        out.append(line)
    return b"\n".join(out)


def scenes():
    out = []
    for key, (turl, purl, granule, epsg) in GRANULES.items():
        tag = key.split()[1]
        d = DATA / "xml"
        d.mkdir(exist_ok=True)
        tile = fetch(f"{S3}/{turl}", d / f"{tag}_MTD_TL.xml")
        product = fetch(f"{S3}/{purl}", d / f"{tag}_MTD_MSIL1C.xml")
        out.append((key, product, tile, granule, epsg))
        if tag == "05.09":
            out.append((key + ", Sentinel-2A", product.replace(b"Sentinel-2B", b"Sentinel-2A"), tile, granule, epsg))
            out.append((key + ", Sentinel-2C", product.replace(b"Sentinel-2B", b"Sentinel-2C"), tile, granule, epsg))
            out.append((key + ", quantification 1000", product.replace(b">10000<", b">1000<"), tile, granule, epsg))
            out.append((key + ", partial tile", product, PARTIAL_TL.read_bytes(), granule, epsg))
            out.append((key + ", valid cells from (8, 5)", product, blank_corner(tile, 5, 8), granule, epsg))
            out.append((key + ", long name", product, tile, "S2B_OPER_MSI_L1C_TL_2BPS_20230728T150149_A033383_T23LKC_N05.09", epsg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=[1100, 800], help="columns and rows of the digital numbers")
    args = ap.parse_args()
    no_power_throttling()
    import zeit

    print("zeit from", Path(zeit.__file__).resolve().parent)
    assert REPO in Path(zeit.__file__).resolve().parents, "not the clone's zeit"
    if not args.skip_build:
        build_harness()
    rng = np.random.default_rng(3110)
    nx, ny = args.size
    dn = synthetic_dn(rng, nx, ny)
    table = Table()
    print(f"{'scene':50s} {'field':14s} {'values':>10s} {'equal':>10s}")
    for n_, (title, product, tile, granule, epsg) in enumerate(scenes()):
        gran = lay_out(f"s{n_}", product, tile, granule, epsg)
        for subset in (False, True):
            for buf in (False, True):
                if buf and not subset:
                    continue
                label = f"{title}, subset={int(subset)}, buffer={int(buf)}"
                # the pixels once per scene and flag set (the metadata does not depend on them)
                c = run_force(f"s{n_}_{int(subset)}{int(buf)}", gran, granule, subset, buf, dn)
                z = run_zeit(product, tile, granule, subset, buf, dn)
                compare(table, label, c, z)
    print(f"{'total':50s} {'':14s} {table.tv:10d} {table.te:10d}")
    sys.exit(1 if table.bad else 0)


if __name__ == "__main__":
    main()
