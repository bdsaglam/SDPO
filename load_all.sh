#!/bin/bash
# Load and split all datasets from HuggingFace into train/test JSON files.
# Run this before preprocess_all.sh.
# Usage: ./load_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Loading datasets ==="

# SciKnowEval
for domain in Biology Chemistry Material Physics; do
    lower=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    output_path="$SCRIPT_DIR/datasets/sciknoweval/$lower/$lower.json"
    output_dir="$SCRIPT_DIR/datasets/sciknoweval/$lower"
    mkdir -p "$output_dir"

    if [ -f "$output_dir/train.json" ] && [ -f "$output_dir/test.json" ]; then
        echo "SKIP SciKnowEval $domain (train.json and test.json already exist)"
    else
        echo "LOADING SciKnowEval $domain ..."
        PYTHONPATH="$SCRIPT_DIR" python "$SCRIPT_DIR/data/load_dataset.py" \
            --dataset_name "$domain" \
            --output_path "$output_path"

        echo "SPLITTING SciKnowEval $domain ..."
        PYTHONPATH="$SCRIPT_DIR" python "$SCRIPT_DIR/data/split_tasks.py" \
            --json_path "$output_path" \
            --output_dir "$output_dir" \
            --test_ratio 0.1 \
            --seed 42
        echo "DONE SciKnowEval $domain"
    fi
done

# LiveCodeBench v6
lcb_json="$SCRIPT_DIR/datasets/lcb_v6.json"
lcb_dir="$SCRIPT_DIR/datasets/lcb_v6"
mkdir -p "$lcb_dir"

if [ -f "$lcb_dir/train.json" ] && [ -f "$lcb_dir/test.json" ]; then
    echo "SKIP LiveCodeBench v6 (train.json and test.json already exist)"
else
    echo "LOADING LiveCodeBench v6 ..."
    PYTHONPATH="$SCRIPT_DIR" python "$SCRIPT_DIR/data/load_dataset.py" \
        --dataset_name livecodebench/code_generation_lite-v6 \
        --output_path "$lcb_json"

    echo "SPLITTING LiveCodeBench v6 ..."
    PYTHONPATH="$SCRIPT_DIR" python "$SCRIPT_DIR/data/split_tests.py" \
        --json_path "$lcb_json" \
        --output_dir "$lcb_dir"
    echo "DONE LiveCodeBench v6"
fi

# Tooluse (already provided, no loading needed)
echo "SKIP Tooluse (pre-provided in datasets/tooluse/)"

echo ""
echo "All loading and splitting complete."
echo "Run ./preprocess_all.sh next to convert to parquet format."
