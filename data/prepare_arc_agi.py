#!/usr/bin/env python3
"""Prepare ARC-AGI dataset for SDPO training.

Loads ARC challenge/solution JSON files, formats them into SDPO schema,
and produces train/test JSON + parquet files.

Usage:
    python data/prepare_arc_agi.py \
        --data_folder /path/to/arc-prize-2024 \
        --output_dir datasets/arc_agi/2024
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data.format.arc_agi import load_arc_agi
from data.preprocess import run_proprocessing
from data.utils.data_handling import write_hf_to_json


def main():
    parser = argparse.ArgumentParser(description="Prepare ARC-AGI dataset for SDPO training.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="datasets/arc_agi/2024/raw",
        help="Path to the directory containing arc-agi_*_challenges.json and arc-agi_*_solutions.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/arc_agi/2024",
        help="Output directory for train/test JSON and parquet files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ARC-AGI data from {args.data_folder}")
    train_ds, eval_ds = load_arc_agi(args.data_folder)

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
    train_path = os.path.join(args.output_dir, "train.json")
    test_path = os.path.join(args.output_dir, "test.json")
    write_hf_to_json(train_ds, train_path)
    write_hf_to_json(eval_ds, test_path)
    print(f"Wrote {train_path} and {test_path}")

    # Run preprocessing to produce parquet
    print("Running preprocessing to produce parquet files...")
    run_proprocessing(args.output_dir, num_proc=1)
    print("Done!")


if __name__ == "__main__":
    main()
