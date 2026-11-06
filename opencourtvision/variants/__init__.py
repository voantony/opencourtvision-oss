"""Build variants (eng notes §2.8.8, CEO decision 18 Oct 04:43).

Common code (what "Vamos US" ships) never names the best-variant package. The non-US "Vamos" artefact contains one extra
sub-package here, which registers itself on discovery; the US artefact is built without it (scripts/build_variant.py)
and CI scans the US artefact for its symbols (scripts/scan_us_build.py, ci/us_build_denylist.txt).
"""
import importlib
import pkgutil

REGISTRY = {}
SCORERS = {}          # name -> scorer callable registered by a variant package (full app: the global decoder)


def discover():
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:
            importlib.import_module(f"{__name__}.{m.name}")
    return REGISTRY


def has_full():
    return bool(discover())
