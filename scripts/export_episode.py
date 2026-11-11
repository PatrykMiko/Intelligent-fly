"""Eksport przykładowego epizodu do JSON (dla odtwarzacza w C++).

    python -m scripts.export_episode --out docs/przyklad_epizodu.json --seed 0 --split train
"""
import argparse

import numpy as np

from task import Splits, TaskConfig, episode_to_dict, make_batch, write_episode_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--screen-encoding", default="onehot_signed")
    ap.add_argument("--screen-noise", type=float, default=0.0)
    a = ap.parse_args()
    cfg = TaskConfig(seq_len=a.seq_len, screen_encoding=a.screen_encoding, screen_noise=a.screen_noise)
    batch = make_batch(Splits(cfg), a.split, 1, cfg, np.random.default_rng(a.seed))
    write_episode_json(episode_to_dict(batch, 0, cfg, a.split), a.out)
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
