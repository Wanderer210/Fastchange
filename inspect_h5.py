#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py


def _fmt_shape(x):
    try:
        return tuple(int(i) for i in x.shape)
    except Exception:
        return None


def _safe_len(x):
    try:
        return int(len(x))
    except Exception:
        return None


def _print_attrs(h5obj, indent: str):
    if not hasattr(h5obj, "attrs"):
        return
    keys = list(h5obj.attrs.keys())
    if not keys:
        return
    print(f"{indent}attrs:")
    for k in keys:
        v = h5obj.attrs[k]
        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="replace")
            except Exception:
                pass
        if isinstance(v, np.ndarray) and v.size > 16:
            print(f"{indent}  - {k}: ndarray shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{indent}  - {k}: {v!r}")


def _dataset_preview(ds: h5py.Dataset, indent: str, max_elems: int = 12):
    shape = _fmt_shape(ds)
    print(f"{indent}dtype={ds.dtype} shape={shape} chunks={ds.chunks} compression={ds.compression}")

    if shape is None or ds.size == 0:
        return

    if ds.dtype.kind in ("i", "u", "f", "b"):
        try:
            if ds.ndim == 1:
                n = min(ds.shape[0], max_elems)
                x = ds[:n]
            elif ds.ndim == 2:
                n0 = min(ds.shape[0], max_elems)
                n1 = min(ds.shape[1], max_elems)
                x = ds[:n0, :n1]
            else:
                slc = tuple(slice(0, min(int(s), 4)) for s in ds.shape)
                x = ds[slc]
            x = np.asarray(x)
            finite = np.isfinite(x) if x.dtype.kind == "f" else np.ones_like(x, dtype=bool)
            if finite.any():
                mn = float(np.min(x[finite]))
                mx = float(np.max(x[finite]))
                mean = float(np.mean(x[finite]))
                nnz = float(np.count_nonzero(x))
                total = float(x.size)
                print(f"{indent}stats: min={mn:.6g} max={mx:.6g} mean={mean:.6g} nonzero={nnz/total:.6g}")
            else:
                print(f"{indent}stats: all values are non-finite in preview slice")
        except Exception as e:
            print(f"{indent}stats: <failed to read preview slice: {e}>")

    if ds.name.endswith("events") or ds.name.split("/")[-1] == "events":
        try:
            n = min(ds.shape[0], 10)
            head = np.asarray(ds[:n])
            print(f"{indent}events head (first {n} rows):")
            print(f"{indent}{head}")
        except Exception as e:
            print(f"{indent}events head: <failed: {e}>")


def _walk(h5obj, path: str = "/", indent: str = ""):
    if isinstance(h5obj, h5py.File):
        print(f"{path} (H5 file)")
        _print_attrs(h5obj, indent + "  ")

    def visitor(name, obj):
        full = f"/{name}" if not name.startswith("/") else name
        if isinstance(obj, h5py.Group):
            print(f"{indent}{full} (group) keys={len(obj.keys())}")
            _print_attrs(obj, indent + "  ")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{full} (dataset)")
            _print_attrs(obj, indent + "  ")
            _dataset_preview(obj, indent + "  ")
        else:
            print(f"{indent}{full} ({type(obj).__name__})")

    h5obj.visititems(visitor)


def _try_common_keys(f: h5py.File):
    common = ["events", "data", "voxel", "voxel_grid", "frames", "timestamps"]
    found = [k for k in common if k in f]
    if not found:
        return
    print("\nCommon top-level datasets:")
    for k in found:
        obj = f[k]
        if isinstance(obj, h5py.Dataset):
            print(f"  - /{k}: dtype={obj.dtype} shape={_fmt_shape(obj)}")
        elif isinstance(obj, h5py.Group):
            print(f"  - /{k}: (group) keys={len(obj.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="path to .h5 file")
    args = ap.parse_args()

    p = Path(args.h5)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    with h5py.File(p, "r") as f:
        _walk(f)
        _try_common_keys(f)


if __name__ == "__main__":
    main()