#!/usr/bin/env python3
"""Prepare ARC-AGI dataset for SDPO training.

Loads ARC challenge/solution JSON files, formats them into SDPO schema,
and produces train/test JSON + parquet files.

If ``arc-agi_training_hints.json`` / ``arc-agi_evaluation_hints.json``
(mapping task_id → analysis text) exist in the data folder, analysis hints
are prepended to prompts and tasks without hints are discarded.

Usage:
    # Without hints (all tasks)
    python data/prepare_arc_agi.py \
        --data_folder datasets/arc_agi/2024/raw \
        --output_dir datasets/arc_agi/2024

    # With hints (place hint files in raw/, output to separate dir)
    python data/prepare_arc_agi.py \
        --data_folder datasets/arc_agi/2024/raw \
        --output_dir datasets/arc_agi/2024-with-hints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.format.arc_agi import load_arc_agi
from data.preprocess import run_proprocessing
from data.utils.data_handling import write_hf_to_json


def _inject_analysis(dataset, analysis: dict):
    """Prepend analysis hints to prompts and filter to tasks with analysis."""
    # Filter to tasks that have analysis entries
    dataset = dataset.filter(lambda ex: ex["task_id"] in analysis)

    def append_hint(ex):
        hint = analysis[ex["task_id"]]
        ex["prompt"] = ex["prompt"] + f"\n\n**Hint:**\n{hint}"
        return ex

    return dataset.map(append_hint)


def main():
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI dataset for SDPO training.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="datasets/arc_agi/2024",
        help="Path to the directory containing arc-agi_*_challenges.json and arc-agi_*_solutions.json files.",
    )
    args = parser.parse_args()

    data_folder = Path(args.data_folder)

    # Auto-detect per-split hint files
    def _load_hints(split_name):
        path = data_folder / "raw" / f"arc-agi_{split_name}_hints.json"
        if path.exists():
            hints = json.loads(path.read_text())
            print(f"Loaded {len(hints)} hints from {path}")
            return hints
        return None

    train_hints = _load_hints("training")
    eval_hints = _load_hints("evaluation")

    print(f"Loading ARC-AGI data from {data_folder}")
    train_ds, eval_ds = load_arc_agi(data_folder / "raw")

    # Inject hints and filter to tasks with hints
    if train_hints is not None:
        before = len(train_ds)
        train_ds = _inject_analysis(train_ds, train_hints)
        print(f"Train after hint filter: {before}->{len(train_ds)}")
    if eval_hints is not None:
        before = len(eval_ds)
        eval_ds = _inject_analysis(eval_ds, eval_hints)
        print(f"Eval after hint filter: {before}->{len(eval_ds)}")

    # Add idx column
    train_ds = train_ds.add_column("idx", list(range(len(train_ds))))
    eval_ds = eval_ds.add_column("idx", list(range(len(eval_ds))))

    # Add embedding column (empty)
    train_ds = train_ds.add_column("embedding", [np.array([])] * len(train_ds))
    eval_ds = eval_ds.add_column("embedding", [np.array([])] * len(eval_ds))

    # Add description suffix
    train_ds = train_ds.map(lambda ex: {"description": ex["description"] + " The solution will be evaluated in a arc_agi environment."})
    eval_ds = eval_ds.map(lambda ex: {"description": ex["description"] + " The solution will be evaluated in a arc_agi environment."})

    # Select final columns
    final_columns = ["idx", "kind", "dataset", "answer", "elo", "prompt", "description", "tests", "embedding", "system"]
    train_ds = train_ds.select_columns([c for c in final_columns if c in train_ds.column_names])
    eval_ds = eval_ds.select_columns([c for c in final_columns if c in eval_ds.column_names])

    print(f"Train: {len(train_ds)} tasks, Eval: {len(eval_ds)} tasks")
    print(f"Columns: {train_ds.column_names}")

    # Write JSON
    train_path = data_folder / "train.json"  
    test_path = data_folder / "test.json"
    write_hf_to_json(train_ds, str(train_path))
    write_hf_to_json(eval_ds, str(test_path))
    print(f"Wrote {train_path} and {test_path}")

    # Run preprocessing to produce parquet
    if len(train_ds) == 0 and len(eval_ds) == 0:
        print("No tasks found — skipping parquet preprocessing.")
    else:
        print("Running preprocessing to produce parquet files...")
        run_proprocessing(str(data_folder), num_proc=1)
        print("Done!")


if __name__ == "__main__":
    main()
