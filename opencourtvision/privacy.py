"""Closed EU beta privacy gates (Vamos Ops checks V2/V7, 21 Oct 2026).

Legal's position for the beta is that processing happens on the device. In beta mode:
- no model is fetched at runtime from Google (storage.googleapis.com, Google Drive) or Hugging Face;
- weights come from the app bundle (``VAMOS_MODEL_DIR``) or the local cache, or, if ``VAMOS_MODEL_HOST`` is set, from our
  own EU host (https only, sha256-pinned where the pin is known);
- ``HF_HUB_OFFLINE=1`` (and the transformers/datasets equivalents) are set for the process;
- a feature whose weights are not local fails closed: it raises ``BetaModelUnavailable`` (or, for optional STT, is
  skipped), it never falls back to a third-party download.

Beta mode is on when ``VAMOS_BETA=1``. The app entry point (``python -m opencourtvision.app``) turns it on by default,
so the shipped path is gated unless a developer sets ``VAMOS_BETA=0`` on purpose. Research scripts are unaffected.
"""
import hashlib
import os
import urllib.parse
import urllib.request

BLOCKED_HOSTS = ("google.com", "googleapis.com", "googleusercontent.com", "gstatic.com", "huggingface.co", "hf.co")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


class BetaModelUnavailable(RuntimeError):
    """Beta mode: the feature's weights are not local and no allowed host serves them, so the feature is disabled."""


def beta_mode(env=None):
    env = os.environ if env is None else env
    return str(env.get("VAMOS_BETA", "")).strip().lower() in ("1", "true", "yes", "on")


def enable_beta(env=None):
    """Turn beta mode on unless VAMOS_BETA is already set; in beta mode force the Hugging Face offline switches."""
    env = os.environ if env is None else env
    env.setdefault("VAMOS_BETA", "1")
    if beta_mode(env):
        env.update(OFFLINE_ENV)
    return beta_mode(env)


def host_of(url):
    return (urllib.parse.urlparse(url).hostname or "").lower()


def is_blocked(url):
    h = host_of(url)
    return any(h == b or h.endswith("." + b) for b in BLOCKED_HOSTS)


def model_host():
    """Our EU model host (VAMOS_MODEL_HOST), or None. Must be https and not a blocked third party."""
    h = os.environ.get("VAMOS_MODEL_HOST", "").strip()
    if not h:
        return None
    if urllib.parse.urlparse(h).scheme != "https" or is_blocked(h):
        raise BetaModelUnavailable(f"VAMOS_MODEL_HOST must be our own https host, got {h!r}")
    return h.rstrip("/")


def bundled(name):
    """Path of `name` inside the app's bundled model dir (VAMOS_MODEL_DIR) if it exists there, else None."""
    d = os.environ.get("VAMOS_MODEL_DIR")
    p = os.path.join(d, name) if d else None
    return p if p and os.path.exists(p) else None


def model_url(name, upstream):
    """Where to fetch `name` from. Outside beta: the upstream URL. In beta: our EU host, or BetaModelUnavailable."""
    if not beta_mode():
        return upstream
    h = model_host()
    if not h:
        raise BetaModelUnavailable(f"{name}: not bundled and no VAMOS_MODEL_HOST set; feature disabled in the beta")
    return f"{h}/{name}"


def check_url(url):
    """Refuse a third-party model fetch in beta mode. Only VAMOS_MODEL_HOST is allowed."""
    if not beta_mode():
        return url
    h = model_host()
    if is_blocked(url) or not h or host_of(url) != host_of(h):
        raise BetaModelUnavailable(f"beta mode: refusing runtime model download from {host_of(url) or url!r}")
    return url


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, sha256=None):
    """The one model download path. Beta mode allows only our EU host; a sha256 pin is checked when given."""
    check_url(url)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    if sha256 and _sha256(tmp) != sha256:
        os.remove(tmp)
        raise RuntimeError(f"{os.path.basename(dest)}: SHA-256 mismatch from {host_of(url)}")
    os.replace(tmp, dest)
    return dest


def require_local(path, feature):
    """Beta mode: `path` must already be on disk, else the feature is disabled (BetaModelUnavailable)."""
    if beta_mode() and not (path and os.path.exists(path)):
        raise BetaModelUnavailable(f"{feature}: weights not on the device ({path}); feature disabled in the beta")
    return path
