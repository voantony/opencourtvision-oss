"""Closed EU beta privacy gates (Vamos Ops V7, 21 Oct 2026): on-device processing only."""
import os
import socket
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from opencourtvision import privacy

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ("google", "googleapis", "huggingface", "hf.co")


@pytest.fixture
def beta(monkeypatch, tmp_path):
    """Beta on, no EU host, empty cache, and every network path recorded and refused."""
    for k in ("VAMOS_MODEL_HOST", "VAMOS_MODEL_DIR", "OCV_WHISPER_MODEL") + tuple(privacy.OFFLINE_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAMOS_BETA", "1")
    monkeypatch.setenv("OCV_CACHE", str(tmp_path / "cache"))
    hits = []

    def record(*a, **k):
        hits.append(str(a[0]) if a else repr(k))
        raise AssertionError(f"network call in beta mode: {a[:1]}")

    monkeypatch.setattr(urllib.request, "urlretrieve", record)
    monkeypatch.setattr(urllib.request, "urlopen", record)
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: record(addr))
    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=lambda **k: record(k.get("id"))))
    return hits


# ---- V7: no runtime model downloads from Google or Hugging Face ----

def test_blocked_hosts():
    for u in ("https://storage.googleapis.com/mediapipe-models/x.task", "https://drive.usercontent.google.com/download?id=1",
              "https://drive.google.com/uc?id=1", "https://huggingface.co/mlx-community/whisper", "https://cdn-lfs.hf.co/x"):
        assert privacy.is_blocked(u), u
    assert not privacy.is_blocked("https://models.vamos.example.eu/mediapipe/x.task")
    assert not privacy.is_blocked("https://notgoogle.com.example.eu/x")


def test_beta_refuses_third_party_even_with_eu_host(beta, monkeypatch):
    monkeypatch.setenv("VAMOS_MODEL_HOST", "https://models.vamos.example.eu")
    for u in ("https://storage.googleapis.com/a.task", "https://huggingface.co/r", "https://other.example.com/a"):
        with pytest.raises(privacy.BetaModelUnavailable):
            privacy.download(u, "/tmp/never")
    assert beta == []


def test_eu_host_must_be_https_and_ours(beta, monkeypatch):
    for bad in ("http://models.vamos.example.eu", "https://huggingface.co"):
        monkeypatch.setenv("VAMOS_MODEL_HOST", bad)
        with pytest.raises(privacy.BetaModelUnavailable):
            privacy.model_url("mediapipe/pose_landmarker_full.task", "https://storage.googleapis.com/x")


def test_beta_eu_host_is_the_only_allowed_download(beta, monkeypatch, tmp_path):
    monkeypatch.setenv("VAMOS_MODEL_HOST", "https://models.vamos.example.eu/")
    got = []
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dest: (got.append(url), Path(dest).write_bytes(b"w")))
    url = privacy.model_url("mediapipe/pose_landmarker_full.task", "https://storage.googleapis.com/x")
    privacy.download(url, str(tmp_path / "m.task"))
    assert got == ["https://models.vamos.example.eu/mediapipe/pose_landmarker_full.task"]
    with pytest.raises(RuntimeError, match="SHA-256"):
        privacy.download(url, str(tmp_path / "m2.task"), sha256="0" * 64)
    assert not (tmp_path / "m2.task").exists()





def _fake_mediapipe(monkeypatch, landmarker="landmarker"):
    """Stub the optional mediapipe import so loader tests exercise privacy, not pip extras."""
    mp = types.ModuleType("mediapipe")
    mp.tasks = types.SimpleNamespace(
        BaseOptions=lambda **k: object(),
        vision=types.SimpleNamespace(
            PoseLandmarkerOptions=lambda **k: object(),
            RunningMode=types.SimpleNamespace(IMAGE="IMAGE"),
            PoseLandmarker=types.SimpleNamespace(create_from_options=lambda opts: landmarker),
        ),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    return mp


def test_pose_loader_fails_closed_in_beta(beta, monkeypatch):
    from opencourtvision.pose import estimate as pe
    _fake_mediapipe(monkeypatch)
    with pytest.raises(privacy.BetaModelUnavailable):
        pe.MediaPipeEstimator()
    assert beta == []


def test_pose_loader_uses_bundled_weights(beta, tmp_path, monkeypatch):
    dest = tmp_path / "weights" / "mediapipe" / "pose_landmarker_full.task"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"fake-pose")
    monkeypatch.setenv("VAMOS_MODEL_DIR", str(tmp_path / "weights"))
    from opencourtvision.pose import estimate as pe
    _fake_mediapipe(monkeypatch)
    est = pe.MediaPipeEstimator()
    assert est._lm == "landmarker"
    assert beta == []


def test_tracknet_fetch_never_uses_google_drive_in_beta(beta, tmp_path):
    from opencourtvision.ball import tracknet3
    with pytest.raises(privacy.BetaModelUnavailable):
        tracknet3.fetch_weights(dest=str(tmp_path / "tn"))
    assert beta == []


def test_tracknet_loader_fails_closed_in_beta(beta):
    pytest.importorskip("torch")
    from opencourtvision.ball import tracknet3
    with pytest.raises(privacy.BetaModelUnavailable):
        tracknet3.TrackNetV3()
    assert beta == []


def test_whisper_stt_off_without_local_weights(beta, monkeypatch, tmp_path):
    from opencourtvision.scoring import callouts
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=lambda *a, **k: pytest.fail("STT ran")))
    assert callouts.stt_model("mlx-community/whisper-large-v3-turbo") is None
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    R = [dict(t0=20.0, t1=25.0)]
    assert callouts.transcribe_callouts(R, 0, "unused.wav", str(tmp_path), run_stt=True) == {}
    assert beta == []
    (tmp_path / "whisper").mkdir(); monkeypatch.setenv("VAMOS_MODEL_DIR", str(tmp_path))
    assert callouts.stt_model("mlx-community/whisper-large-v3-turbo") == str(tmp_path / "whisper")


def test_outside_beta_behaviour_unchanged(monkeypatch):
    monkeypatch.delenv("VAMOS_BETA", raising=False)
    from opencourtvision.scoring import callouts
    assert privacy.model_url("x", "https://storage.googleapis.com/x") == "https://storage.googleapis.com/x"
    assert callouts.stt_model("mlx-community/whisper-large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_app_entry_turns_beta_and_hf_offline_on():
    env = {k: v for k, v in os.environ.items() if k not in ("VAMOS_BETA",) + tuple(privacy.OFFLINE_ENV)}
    code = "import os, opencourtvision.app.__main__; print(os.environ['VAMOS_BETA'], os.environ['HF_HUB_OFFLINE'])"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "1"]


def test_only_privacy_module_downloads():
    """Static guard: every model fetch goes through privacy.download (gdown only behind the beta check in tracknet3)."""
    bad = []
    for f in (ROOT / "opencourtvision").rglob("*.py"):
        if f.name == "privacy.py" or "_vendor" in f.parts:     # vendored code is byte-identical upstream; guarded at
            continue                                         # its package __init__ and by runtime tests (PR #26)
        s = f.read_text()
        for needle in ("urlretrieve(", "urlopen(", "hf_hub_download", "snapshot_download", "load_state_dict_from_url", "torch.hub.load"):
            if needle in s:
                bad.append(f"{f.relative_to(ROOT)}: {needle}")
        if "gdown" in s and f.name != "tracknet3.py":
            bad.append(f"{f.relative_to(ROOT)}: gdown")
    assert bad == []
