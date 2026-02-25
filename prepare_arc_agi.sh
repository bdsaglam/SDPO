#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(
    "datasets/arc_agi/dummy"
    "datasets/arc_agi/2024"
    "datasets/arc_agi/2024-with-hints"
    "datasets/arc_agi/2025"
)

for ds in "${DATASETS[@]}"; do
    path="$SCRIPT_DIR/$ds"
    echo "PREPROCESSING $ds ..."
    python "$SCRIPT_DIR/data/prepare_arc_agi.py" --data_folder "$path"
    echo "DONE $ds"
done

echo ""
echo "All preprocessing complete."
