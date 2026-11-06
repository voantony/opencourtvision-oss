import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from opencourtvision.synthetic import make_clip, scenario  # noqa: E402


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    return {n: make_clip(scenario(n), str(d)) for n in ("hits_clean", "match_sim")}
