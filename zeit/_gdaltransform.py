# SPDX-License-Identifier: GPL-2.0-or-later
"""GDAL's coordinate transformer, from the GDAL library rasterio loads (through ctypes).

From landschaft 1.35.0 (_gdaltransform.py, by the same author).

The warp of load_raster(..., like=) computes the source coordinates of destination cells
itself; to be GDAL's to the last bit they must come from GDAL's own transformer
(GDALGenImgProjTransform), with the
PROJ rasterio ships, which may not be pyproj's (their inverse LAEA differ by up to a
millimetre between PROJ 9.5 and 9.8). rasterio does not expose that transformer, and its
point transformation loops in Python, so the functions are called from the GDAL library
already loaded in the process. ctypes releases the GIL during the calls: one transformer
per thread transforms points in parallel. ``open_library()`` returns None when the library
cannot be found; the caller then uses pyproj.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
import threading

import numpy as np

__all__ = ["GdalTransformer", "open_library"]

_lib = None
_lib_lock = threading.Lock()
_searched = False


def _loaded_paths() -> list[str]:
    """Paths of the shared libraries loaded in this process."""
    paths: list[str] = []
    try:
        if sys.platform == "win32":
            from ctypes import wintypes

            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                                 ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
            process = kernel32.GetCurrentProcess()
            modules = (wintypes.HMODULE * 4096)()
            needed = wintypes.DWORD()
            if psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
                count = min(needed.value // ctypes.sizeof(wintypes.HMODULE), len(modules))
                buf = ctypes.create_unicode_buffer(32768)
                for h in modules[:count]:
                    if kernel32.GetModuleFileNameW(h, buf, len(buf)):
                        paths.append(buf.value)
        elif sys.platform == "darwin":
            libc = ctypes.CDLL(None)
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            for i in range(libc._dyld_image_count()):
                name = libc._dyld_get_image_name(i)
                if name:
                    paths.append(name.decode("utf-8", "replace"))
        else:
            with open("/proc/self/maps") as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        paths.append(parts[5].strip())
    except Exception:  # noqa: BLE001, S110 - any failure: fall back to searching the files
        pass
    return paths


def _candidates() -> list[str]:
    import rasterio
    import rasterio._base  # loads rasterio's GDAL

    def is_gdal(path):
        name = os.path.basename(path).lower()
        return name.startswith(("gdal", "libgdal")) and (".dll" in name or ".so" in name or ".dylib" in name)

    found = [p for p in dict.fromkeys(_loaded_paths()) if is_gdal(p)]
    root = os.path.dirname(rasterio.__file__)
    for pattern in (os.path.join(root + ".libs", "*gdal*"), os.path.join(root, ".dylibs", "*gdal*"),
                    os.path.join(root, "*gdal*"), os.path.join(sys.prefix, "Library", "bin", "gdal*.dll"),
                    os.path.join(sys.prefix, "lib", "libgdal*")):
        found += [p for p in glob.glob(pattern) if is_gdal(p) and p not in found]
    return found


def _bind(lib) -> None:
    c_void_p, c_char_p, c_int, c_double = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double
    p_double = ctypes.POINTER(c_double)
    p_int = ctypes.POINTER(c_int)
    lib.GDALVersionInfo.argtypes = [c_char_p]
    lib.GDALVersionInfo.restype = c_char_p
    lib.GDALAllRegister.argtypes = []
    lib.GDALAllRegister.restype = None
    lib.GDALGetDriverByName.argtypes = [c_char_p]
    lib.GDALGetDriverByName.restype = c_void_p
    lib.GDALCreate.argtypes = [c_void_p, c_char_p, c_int, c_int, c_int, c_int, ctypes.POINTER(c_char_p)]
    lib.GDALCreate.restype = c_void_p
    lib.GDALSetGeoTransform.argtypes = [c_void_p, p_double]
    lib.GDALSetGeoTransform.restype = c_int
    lib.GDALSetProjection.argtypes = [c_void_p, c_char_p]
    lib.GDALSetProjection.restype = c_int
    lib.GDALClose.argtypes = [c_void_p]
    lib.GDALClose.restype = None
    lib.GDALCreateGenImgProjTransformer2.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_char_p)]
    lib.GDALCreateGenImgProjTransformer2.restype = c_void_p
    lib.GDALGenImgProjTransform.argtypes = [c_void_p, c_int, c_int, p_double, p_double, p_double, p_int]
    lib.GDALGenImgProjTransform.restype = c_int
    lib.GDALDestroyGenImgProjTransformer.argtypes = [c_void_p]
    lib.GDALDestroyGenImgProjTransformer.restype = None


def open_library():
    """rasterio's GDAL library with the functions bound, or None."""
    global _lib, _searched
    with _lib_lock:
        if _searched:
            return _lib
        _searched = True
        if os.environ.get("ZEIT_WARP_PYPROJ"):
            return None
        import rasterio

        for path in _candidates():
            try:
                lib = ctypes.CDLL(path)
                _bind(lib)
                version = lib.GDALVersionInfo(b"RELEASE_NAME").decode()
            except (OSError, AttributeError):
                continue
            if version == rasterio.__gdal_version__:
                lib.GDALAllRegister()
                _lib = lib
                break
        return _lib


class GdalTransformer:
    """GDAL's GDALGenImgProjTransform from destination pixel/line to source pixel/line, set up
    as rasterio's reproject sets it up (GDALCreateGenImgProjTransformer2 between datasets with
    the two grids, GCPS_OK=TRUE). Each call takes a transformer of its own from a pool (made
    when none is free), so threads transform in parallel. Failed points are infinite."""

    def __init__(self, lib, src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape):
        self.lib = lib
        self._lock = threading.Lock()
        self._free: list[int] = []
        self._made: list[int] = []
        driver = lib.GDALGetDriverByName(b"MEM")
        if not driver:
            raise RuntimeError("GDAL's MEM driver is not available")
        self._datasets = []
        for t, crs, (h, w) in ((src_t, src_crs, src_shape), (dst_t, dst_crs, dst_shape)):
            ds = lib.GDALCreate(driver, b"", int(w), int(h), 0, 1, None)
            if not ds:
                self.close()
                raise RuntimeError("cannot create a GDAL dataset in memory")
            self._datasets.append(ds)
            gt = (ctypes.c_double * 6)(t.c, t.a, t.b, t.f, t.d, t.e)
            lib.GDALSetGeoTransform(ds, gt)
            lib.GDALSetProjection(ds, crs.to_wkt().encode())
        self._options = (ctypes.c_char_p * 2)(b"GCPS_OK=TRUE", None)
        self._release(self._take())  # fail now rather than in a worker thread

    @property
    def address(self) -> int:
        """The address of GDALGenImgProjTransform (a GDALTransformerFunc)."""
        return ctypes.cast(self.lib.GDALGenImgProjTransform, ctypes.c_void_p).value

    def take(self, n: int = 1) -> list[int]:
        """n transformers of their own (made when none is free); give them back with release()."""
        out = []
        with self._lock:
            while len(out) < n:
                if self._free:
                    out.append(self._free.pop())
                    continue
                arg = self.lib.GDALCreateGenImgProjTransformer2(self._datasets[0], self._datasets[1], self._options)
                if not arg:
                    raise RuntimeError("GDAL cannot transform between these coordinate systems")
                self._made.append(arg)
                out.append(arg)
        return out

    def release(self, args) -> None:
        with self._lock:
            self._free.extend(args)

    def _take(self):
        return self.take(1)[0]

    def _release(self, arg):
        self.release([arg])

    def __call__(self, u, v, dst_to_src: bool = True):
        x = np.array(u, dtype=np.float64, copy=True, order="C").ravel()
        y = np.array(np.broadcast_to(v, np.shape(u)), dtype=np.float64, copy=True, order="C").ravel()
        n = x.size
        if n == 0:
            return x, y
        z = np.zeros(n)
        ok = np.zeros(n, dtype=np.intc)
        p_double = ctypes.POINTER(ctypes.c_double)
        arg = self._take()
        try:
            self.lib.GDALGenImgProjTransform(arg, 1 if dst_to_src else 0, n, x.ctypes.data_as(p_double),
                                             y.ctypes.data_as(p_double), z.ctypes.data_as(p_double),
                                             ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        finally:
            self._release(arg)
        bad = ok == 0
        if bad.any():
            x[bad] = np.inf
            y[bad] = np.inf
        return x, y

    def close(self):
        with self._lock:
            for arg in self._made:
                self.lib.GDALDestroyGenImgProjTransformer(arg)
            self._made = []
            self._free = []
            for ds in self._datasets:
                self.lib.GDALClose(ds)
            self._datasets = []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - at interpreter exit
            pass
