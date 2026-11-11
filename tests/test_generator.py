"""Testy generatora zadania (tylko numpy). Uruchom: pytest tests/test_generator.py"""
import numpy as np

from task import (ADD, SUB, MOD, Splits, TaskConfig, all_triples, apply_op, carries, encode_screen,
                  from_bits, make_batch, to_bits)
from task.generator import MOVE_X


def test_bits_roundtrip_all_64():
    n = np.arange(MOD)
    assert np.array_equal(from_bits(to_bits(n)), n)
    # bit i = noga i, wartość 2^i
    assert to_bits(1).tolist() == [1, 0, 0, 0, 0, 0]
    assert to_bits(32).tolist() == [0, 0, 0, 0, 0, 1]
    assert to_bits(13).tolist() == [1, 0, 1, 1, 0, 0]


def test_apply_op_all_4096_pairs_both_ops():
    s, o, x = all_triples()
    assert s.size == 64 * 64 * 2
    got = apply_op(s, o, x)
    for si, oi, xi, g in zip(s, o, x, got):
        want = (int(si) + int(oi) * int(xi)) % 64      # niezależne obliczenie w czystym Pythonie
        assert g == want


def test_result_bit_is_xor_of_state_bit_x_bit_and_carry():
    s, o, x = all_triples()
    res = to_bits(apply_op(s, o, x))
    xor = to_bits(s) ^ to_bits(x) ^ carries(s, o, x)
    assert np.array_equal(res, xor)


def test_carries_known_values():
    # 1 + 1 = 2: przeniesienie z bitu 0 do bitu 1
    c = carries(1, ADD, 1)
    assert c.tolist() == [0, 1, 0, 0, 0, 0]
    # 0 - 1 = 63: pożyczka propaguje się przez wszystkie bity
    c = carries(0, SUB, 1)
    assert c.tolist() == [0, 1, 1, 1, 1, 1]
    # bit 0 nigdy nie ma przeniesienia
    s, o, x = all_triples()
    assert not carries(s, o, x)[:, 0].any()


def test_screen_onehot_signed():
    scr = encode_screen(ADD, 13)
    assert scr.shape == (64,) and scr[13] == 1 and scr.sum() == 1
    scr = encode_screen(SUB, 13)
    assert scr[13] == -1 and np.abs(scr).sum() == 1
    # różne instrukcje -> różne ekrany (poza X=0, gdzie +0 i -0 dają ten sam wynik)
    s, o, x = all_triples()
    scr = encode_screen(o, x).reshape(len(o), -1)
    keys = {tuple(r) for r in scr}
    assert len(keys) == 64 * 2


def test_screen_binary():
    scr = encode_screen(SUB, 13, "binary")
    assert scr[:6].tolist() == [1, 0, 1, 1, 0, 0] and scr[6] == 1 and scr[7:].sum() == 0
    assert encode_screen(ADD, 13, "binary")[6] == 0


def test_splits_are_disjoint_and_complete():
    cfg = TaskConfig()
    sp = Splits(cfg)
    tr, va, te = sp.pair_train, sp.pair_val, sp.pair_test
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert tr.sum() > 2000 and va.sum() > 100 and te.sum() > 200
    # wyłączone stany i X nigdy nie trafiają do puli par treningowych
    assert not tr[sp.held_states, :].any()
    assert not tr[:, sp.held_x].any()
    # w treningu stan docelowy też nie jest wyłączonym stanem
    for s, m in zip(*np.nonzero(sp.criteria["train"])):
        assert sp.next_state[s, m] not in set(sp.held_states.tolist())


def test_train_batches_never_touch_heldout_data():
    cfg = TaskConfig(split_seed=3)
    sp = Splits(cfg)
    b = make_batch(sp, "train", 400, cfg, np.random.default_rng(0))
    assert b["measure"].all()
    held_s, held_x = set(sp.held_states.tolist()), set(sp.held_x.tolist())
    assert not (set(np.unique(b["state_prev"])) & held_s)
    assert not (set(np.unique(b["state_next"])) & held_s)
    assert not (set(np.unique(b["x"])) & held_x)
    # żadna para (stan, X) z val/test nie pojawia się w treningu
    pairs = set(zip(b["state_prev"].ravel().tolist(), b["x"].ravel().tolist()))
    for P in (sp.pair_val, sp.pair_test):
        held = set(zip(*[a.tolist() for a in np.nonzero(P)]))
        assert not (pairs & held)


def test_eval_batches_measure_the_right_things():
    cfg = TaskConfig(split_seed=1)
    sp = Splits(cfg)
    rng = np.random.default_rng(5)
    for split in ("val", "test_pairs", "test_x", "test_states"):
        b = make_batch(sp, split, 200, cfg, rng)
        assert b["measure"].any(), split
        s, x, m = b["state_prev"][b["measure"]], b["x"][b["measure"]], b["measure"]
        if split == "test_x":
            assert set(np.unique(x)) <= set(sp.held_x.tolist())
        if split == "test_states":
            assert set(np.unique(s)) <= set(sp.held_states.tolist())
        if split in ("val", "test_pairs"):
            P = sp.pair_val if split == "val" else sp.pair_test
            assert P[s, x].all()


def test_chain_is_consistent_and_targets_correct():
    cfg = TaskConfig(seq_len=16)
    sp = Splits(cfg)
    b = make_batch(sp, "train", 50, cfg, np.random.default_rng(2))
    # stan_prev[t+1] == stan_next[t]
    assert np.array_equal(b["state_prev"][:, 1:], b["state_next"][:, :-1])
    assert np.array_equal(b["state_next"], apply_op(b["state_prev"], b["op"], b["x"]))
    assert np.array_equal(from_bits(b["targets"].astype(int)), b["state_next"])
    assert b["screens"].shape == (50, 16, 64) and b["legs_in"].shape == (50, 16, 6)


def test_legs_modes():
    sp_cfg = TaskConfig(legs_mode="teacher")
    sp = Splits(sp_cfg)
    b = make_batch(sp, "train", 20, sp_cfg, np.random.default_rng(0))
    assert np.array_equal(from_bits(b["legs_in"].astype(int)), b["state_prev"])
    cfg2 = TaskConfig(legs_mode="init_only")
    b2 = make_batch(Splits(cfg2), "train", 20, cfg2, np.random.default_rng(0))
    assert (b2["legs_in"][:, 1:] == 0).all()
    assert np.array_equal(from_bits(b2["legs_in"][:, 0].astype(int)), b2["state_prev"][:, 0])


def test_noise_changes_screen_only():
    cfg = TaskConfig(screen_noise=0.3)
    sp = Splits(cfg)
    b = make_batch(sp, "train", 10, cfg, np.random.default_rng(0))
    clean = encode_screen(b["op"], b["x"])
    assert not np.array_equal(b["screens"], clean)
    assert abs((b["screens"] - clean).std() - 0.3) < 0.05


def test_reproducible_with_same_seed():
    cfg = TaskConfig()
    sp = Splits(cfg)
    a = make_batch(sp, "train", 8, cfg, np.random.default_rng(7))
    b = make_batch(sp, "train", 8, cfg, np.random.default_rng(7))
    assert np.array_equal(a["state_prev"], b["state_prev"]) and np.array_equal(a["x"], b["x"])
