"""Ops V7 (closed EU beta): OSNet weights are bundled or come from our EU host; Google Drive is never contacted."""
import socket
import sys
import urllib.request

import pytest

from opencourtvision import privacy


@pytest.fixture
def beta(monkeypatch, tmp_path):
    for k in ("VAMOS_MODEL_HOST", "VAMOS_MODEL_DIR"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VAMOS_BETA", "1"); monkeypatch.setenv("OCV_CACHE", str(tmp_path / "cache"))
    fail = lambda *a, **k: pytest.fail(f"network call in beta mode: {a[:1]}")  # noqa: E731
    monkeypatch.setattr(urllib.request, "urlretrieve", fail)
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setitem(sys.modules, "gdown", None)
    return tmp_path


def test_osnet_weights_never_from_google_in_beta(beta):
    from opencourtvision.players import reid
    with pytest.raises(privacy.BetaModelUnavailable):
        reid.weights_path()


def test_osnet_bundled_weights_used(beta, monkeypatch):
    from opencourtvision.players import reid
    (beta / "osnet").mkdir(); (beta / "osnet" / "osnet_x1_0_msmt17.pth").write_bytes(b"w")
    monkeypatch.setenv("VAMOS_MODEL_DIR", str(beta))
    assert reid.weights_path() == str(beta / "osnet" / "osnet_x1_0_msmt17.pth")


def test_vendored_osnet_imagenet_download_blocked_in_beta(beta):
    pytest.importorskip("torch")
    from opencourtvision.players._vendor.osnet import osnet as O
    with pytest.raises(privacy.BetaModelUnavailable):
        O.osnet_x0_25(num_classes=1, pretrained=True)
