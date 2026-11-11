"""Testy grafów (tylko numpy). Uruchom: pytest tests/test_graph.py"""
import numpy as np

from graphs import CONTROLS, apply_control, make_random_layered


def _g():
    return make_random_layered(seed=0, n_hidden=200, n_input=32, n_sensor=12, motor_per_leg=4)


def test_graph_basic_shape_and_roles():
    g = _g()
    assert g.n == 32 + 12 + 3 * 200 + 24
    assert g.n_edges > 1000
    assert len(g.motor_idx) == 6 and all(len(m) == 4 for m in g.motor_idx)
    roles = np.concatenate([g.input_idx, g.sensor_idx] + g.motor_idx)
    assert len(np.unique(roles)) == len(roles)                       # role się nie nakładają
    assert g.src.max() < g.n and g.dst.max() < g.n


def test_normalization_and_dale():
    g = _g()
    tot = np.zeros(g.n)
    np.add.at(tot, g.dst, g.mag)
    has_in = g.in_degree() > 0
    assert np.allclose(tot[has_in], 1.0, atol=1e-5)
    for i in np.unique(g.src)[:50]:                                   # znak zależy tylko od src
        assert len(np.unique(g.sign[g.src == i])) == 1
    assert 0.1 < g.inhibitory_fraction() < 0.3


def test_degree_preserving_control():
    g = _g()
    c = apply_control(g, "degree_preserving", seed=1)
    assert np.array_equal(np.sort(c.out_degree()), np.sort(g.out_degree()))
    assert np.array_equal(c.out_degree(), g.out_degree())
    assert np.array_equal(np.sort(c.in_degree()), np.sort(g.in_degree()))
    assert not np.array_equal(c.dst, g.dst)
    assert np.array_equal(c.sign, g.sign)


def test_shuffled_weights_keeps_topology():
    g = _g()
    c = apply_control(g, "shuffled_weights", seed=1)
    assert np.array_equal(c.src, g.src) and np.array_equal(c.dst, g.dst)
    assert not np.array_equal(c.sign, g.sign)
    assert abs(c.inhibitory_fraction() - g.inhibitory_fraction()) < 1e-6   # ta sama frakcja, inne miejsca


def test_no_signs_and_random_signs():
    g = _g()
    assert (apply_control(g, "no_signs").sign == 1).all()
    r = apply_control(g, "random_signs", seed=3)
    assert abs(r.inhibitory_fraction() - g.inhibitory_fraction()) < 0.08
    assert not np.array_equal(r.sign, g.sign)
    assert np.array_equal(r.src, g.src) and np.array_equal(r.dst, g.dst)


def test_controls_do_not_modify_original():
    g = _g()
    before = (g.dst.copy(), g.sign.copy(), g.mag.copy())
    for k in CONTROLS:
        apply_control(g, k, seed=0)
    assert np.array_equal(before[0], g.dst) and np.array_equal(before[1], g.sign) and np.array_equal(before[2], g.mag)


def test_save_load_roundtrip(tmp_path=None):
    import tempfile, os
    g = _g()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.npz")
        g.save(p)
        from graphs import GraphSpec
        h = GraphSpec.load(p)
    assert h.n == g.n and np.array_equal(h.src, g.src) and np.array_equal(h.sign, g.sign)
    assert all(np.array_equal(a, b) for a, b in zip(h.motor_idx, g.motor_idx))


def test_reproducible_graph():
    a, b = make_random_layered(seed=5, n_hidden=100), make_random_layered(seed=5, n_hidden=100)
    assert np.array_equal(a.src, b.src) and np.array_equal(a.dst, b.dst)


def test_spectral_radius_matches_dense_eigenvalues():
    from graphs import spectral_radius
    g = make_random_layered(seed=2, n_hidden=150, n_input=24, n_sensor=8, motor_per_leg=4)
    W = np.zeros((g.n, g.n))
    np.add.at(W, (g.dst, g.src), g.sign * g.mag)
    exact = np.abs(np.linalg.eigvals(W)).max()
    est = spectral_radius(g, iters=400)
    assert abs(est - exact) / exact < 0.1, (est, exact)
    # skalowanie wag skaluje promień (tu: usunięcie znaków zwiększa lub zmienia promień, ale zawsze > 0)
    assert spectral_radius(apply_control(g, "no_signs")) > 0
