#!/usr/bin/env python3
"""Fail if a "Vamos US" artefact contains best-variant symbols (ci/us_build_denylist.txt; CI-only, never packaged).

  python scripts/scan_us_build.py dist/us                 # directory, .whl/.zip/.ipa, or a Mach-O / .app binary

Scans every file's bytes (source, bytecode, binaries, dSYM-less app binaries alike), the same way `strings | grep` would.
For iOS, CI runs it on the US .ipa (and on `nm -gU` output of the app binary). Exit 1 and list hits if any symbol appears.
"""
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SYMS = os.path.join(ROOT, "ci", "us_build_denylist.txt")


def symbols(path=SYMS):
    return [s.strip() for s in open(path) if s.strip() and not s.lstrip().startswith("#")]


def _blobs(target):
    if os.path.isdir(target):
        for d, _, fs in os.walk(target):
            for f in fs:
                p = os.path.join(d, f)
                yield os.path.relpath(p, target), open(p, "rb").read()
    elif zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            for n in z.namelist():
                yield n, z.read(n)
    else:
        yield os.path.basename(target), open(target, "rb").read()


def scan(target, syms=None):
    syms = syms or symbols()
    hits = []
    for name, data in _blobs(target):
        hay = name.encode() + b"\n" + data
        hits += [(name, s) for s in syms if s.encode() in hay]
    return hits


def main():
    if len(sys.argv) != 2: sys.exit(__doc__)
    hits = scan(sys.argv[1])
    for n, s in hits: print(f"BEST-VARIANT SYMBOL in US artefact: {s!r} in {n}")
    print(f"scanned {sys.argv[1]}: {len(hits)} hit(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
