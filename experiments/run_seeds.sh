#!/usr/bin/env bash
# Uruchamia config dla ziaren 0..4 (krok 21). Użycie: bash experiments/run_seeds.sh gru [dodatkowe --set ...]
set -euo pipefail
name="$1"; shift
for seed in 0 1 2 3 4; do
  python -m train.train --config "experiments/configs/${name}.yaml" --set seed=$seed "$@"
done
