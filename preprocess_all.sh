#!/bin/bash
# Preprocess all available datasets from JSON to Parquet format.
# Usage: ./preprocess_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(
    "datasets/sciknoweval/biology"
    "datasets/sciknoweval/chemistry"
    "datasets/sciknoweval/material"
    "datasets/sciknoweval/physics"
    "datasets/tooluse"
    "datasets/lcb_v6"
)

for ds in "${DATASETS[@]}"; do
    path="$SCRIPT_DIR/$ds"
    if [ -f "$path/train.json" ] && [ -f "$path/test.json" ]; then
        if [ -f "$path/train.parquet" ] && [ -f "$path/test.parquet" ]; then
            echo "SKIP $ds (parquet files already exist)"
        else
            echo "PREPROCESSING $ds ..."
            python "$SCRIPT_DIR/data/preprocess.py" --data_source "$path"
            echo "DONE $ds"
        fi
    else
        echo "SKIP $ds (train.json or test.json missing)"
    fi
done

echo ""
echo "All preprocessing complete."
