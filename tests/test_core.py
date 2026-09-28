import numpy as np
import pytest

from cspredict.build import load_models, model_dir
from cspredict.dataset import _hash_split, list_demos
from cspredict.filters import DEFAULT_CONFIGS, gather_evidence, run_filter
from cspredict.infostate import episodes
from cspredict.motion import N_SINCE_BINS, since_bin
from cspredict.visibility import _segment_point_dist


def test_segment_point_dist():
    a = np.array([0.0, 0.0])
    b = np.array([[10.0, 0.0], [0.0, 10.0]])
    np.testing.assert_allclose(_segment_point_dist(a, b, np.array([5.0, 3.0])), [3.0, 5.0])
    # Beyond the segment end the distance is to the endpoint.
    np.testing.assert_allclose(_segment_point_dist(a, b[:1], np.array([13.0, 4.0])), [5.0])


def test_since_bins():
    assert since_bin(0.0) == 0
    assert since_bin(2.0) == 1
    assert since_bin(25.0) == N_SINCE_BINS - 2
    assert since_bin(np.inf) == since_bin(np.nan) == N_SINCE_BINS - 1


def test_hash_split_is_stable():
    assert _hash_split("some-demo") == _hash_split("some-demo")
    splits = [_hash_split(f"demo-{i}") for i in range(2000)]
    assert 0.15 < splits.count("test") / 2000 < 0.25


@pytest.fixture(scope="module")
def models():
    for sources in (["hltv", "xego"], ["xego"]):
        path = model_dir(sources)
        if (path / "library.npz").exists():
            return load_models(path)
    pytest.skip("models not built; run `python -m cspredict.build`")


def test_grid_locate_roundtrip(models):
    g = models.grid
    nodes = g.locate(g.node_xy[:, 0], g.node_xy[:, 1], g.node_z + 1.0)
    assert (nodes == np.arange(g.n)).mean() > 0.97


def test_beliefs_are_distributions_and_collapse_on_sightings(models):
    refs = list_demos(None, ["test"])
    if not refs:
        pytest.skip("no parsed test demos")
    ep = next(episodes(models.grid, refs[0], "ct"))
    ev = gather_evidence(ep, models.grid, models.spot, models.fires)
    for cfg in DEFAULT_CONFIGS:
        for s, b in run_filter(ep, models.grid, models.motion, cfg, ev, models.library, models.teams, models.calibration):
            assert b.shape == (len(ep.enemy_ids), models.grid.n)
            assert (b >= 0).all()
            np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-6)
            seen = np.flatnonzero(ep.enemy_seen_node[s] >= 0)
            if cfg.name in ("hmm", "diffuse", "last_seen") and len(seen):
                e = seen[0]
                assert b[e, ep.enemy_seen_node[s, e]] > 0.99
