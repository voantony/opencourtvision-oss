"""OSNet (deep-person-reid, MIT) kept byte-identical to upstream KaiyangZhou/deep-person-reid@f8cd150 torchreid/models/osnet.py.

Our only addition lives here, not in osnet.py: in beta mode the upstream ImageNet download (`init_pretrained_weights`, a
Google Drive fetch via gdown) is replaced by a guard that disables it (Ops V7). reid.py always passes pretrained=False.
"""


def _guard_pretrained_download():
    import importlib

    from .... import privacy
    try:
        mod = importlib.import_module(f"{__name__}.osnet")
    except ImportError:                                  # torch missing: nothing to guard
        return
    upstream = mod.init_pretrained_weights

    def init_pretrained_weights(model, key=""):
        if privacy.beta_mode():
            raise privacy.BetaModelUnavailable("OSNet ImageNet weights are a Google Drive download; disabled in the beta")
        return upstream(model, key)

    mod.init_pretrained_weights = init_pretrained_weights


_guard_pretrained_download()
