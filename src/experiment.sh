#!/bin/bash

set -euo pipefail

DATA_PATH="${1:-../data/Evaluation_CoTs/sample_data/new_extracted_data/test.json}"
SCORE_MODE="${2:-custom}"

thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
buffer_sizes=(1 2 3 4 5 6 7 8 9 10)

for threshold in "${thresholds[@]}"; do
    for buffer_size in "${buffer_sizes[@]}"; do
        python3 CS_based_early_stopping.py \
            --data_path "$DATA_PATH" \
            --threshold "$threshold" \
            --buffer_size "$buffer_size" \
            --score_mode "$SCORE_MODE"
    done
done
