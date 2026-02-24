"""ARC-AGI dataset loader for SDPO training.

Produces the standard SDPO schema (idx, kind, dataset, prompt, answer, tests,
description, elo, system) from ARC-AGI challenge/solution JSON files.

Analysis hints are NOT included here — they are injected dynamically during
training via the arc_analysis hook.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


# ---------------------------------------------------------------------------
# Grid formatting (from rlvr data.py)
# ---------------------------------------------------------------------------

Grid = list[list[int]]


def format_grid(grid: Grid) -> str:
    """Format grid as comma-separated digits per row."""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def format_task_question(train_pairs: list[dict], test_inputs: list[Grid]) -> str:
    """Format an ARC task as a question string with grids."""
    parts: list[str] = []
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"Example #{i}")
        parts.append("Input:")
        parts.append(format_grid(pair["input"]))
        parts.append("Output:")
        parts.append(format_grid(pair["output"]))
        parts.append("")

    for i, inp in enumerate(test_inputs, 1):
        parts.append(f"Challenge #{i}")
        parts.append("Input:")
        parts.append(format_grid(inp))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt (from rlvr envs/repl.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at solving Abstract Reasoning Corpus (ARC) tasks. Your goal is to analyze
input-output examples and produce correct output predictions for the test inputs.

**1. Analyze the Examples:**
  * Identify key objects in input/output grids (shapes, lines, regions) using `scipy.ndimage.label` etc.
  * Determine relationships between objects (spatial arrangement, color, size).
  * Identify operations that transform input to output (rotation, reflection, color change, addition/removal).
  * Consider grid dimensions, symmetries, and other visual features.

**2. Formulate a Hypothesis:**
  * Based on your analysis, formulate a transformation rule that works across all examples.
  * Prioritize simpler rules first.
  * **Generalisation Check:** Will your rule generalise to the test inputs?
  * **Generalisation Advice:**
    * **Orientation/Direction/Shape:** Ensure your hypothesis covers symmetric cases.
    * **Avoid Arbitrary Constants:** Don't rely on constants tuned to training examples (thresholds, offsets, dimensions).
  * Consider these transformation types:
    * **Object Manipulation:** Moving, rotating, reflecting, or resizing objects.
    * **Color Changes:** Changing colors of specific objects or regions.
    * **Spatial Arrangements:** Rearranging objects in specific patterns.
    * **Object Addition/Removal/Swapping:** Based on certain criteria.
    * **Global vs. Local:** Consider whether components are global or local.

**3. Construct Outputs Directly:**

Once you figured out the transformation rule in the puzzle, apply it to the test inputs to get the output grids.
You may write a general `transform()` or `solve()` function when the rule is simple and clearly algorithmic.
However, for visually complex puzzles, a complete general transformation function can be hard to implement robustly in pure Python.
In those cases, build each output grid directly through step-by-step edits in the REPL.

**Why?** Visual pattern recognition (e.g., locating objects, understanding spatial relationships,
or identifying enclosed regions) is often easier by inspection than by writing full detection algorithms
(flood-fill, connected components, complex boundary logic). In those cases, use this loop:
  * **Inspect** printed grids and identify key regions/objects/patterns. You can zoom in on some parts of the grid to see more details using slicing.
  * **Edit/Apply** targeted numpy operations to construct the output
  * **Inspect again** to verify and iterate until the grid matches the rule

Example workflow:
[[ ## reasoning ## ]]
<Your step-by-step reasoning>

[[ ## code ## ]]
```python
# Start with a copy of the input (or create fresh grid)
pred = np.array(task["test"][0]["input"]).copy()

# Make targeted edits based on what you observe
pred[2:5, 3:6] = 4  # Fill region you identified visually
pred[0, :] = 1      # Change top row

# Print to verify
print(format_grid(pred))
```

**4. Test on Training Examples:**

Before submitting, verify your approach works on training examples:
[[ ## code ## ]]
```python
# Apply same strategy to a training input
train_in = np.array(task["train"][0]["input"])
train_pred = train_in.copy()
# ... apply your edits ...
print(f"Accuracy: {accuracy(train_pred, task['train'][0]['output'])}")
```

**5. Submit Predictions:**

Once verified, apply your approach to all test inputs and submit:
[[ ## code ## ]]
```python
SUBMIT(test=[pred_0, pred_1, ...])  # List of output grids
```

---

**Your Environment -- Interactive REPL:**

You work in a persistent Python REPL. Variables persist across iterations.
Pre-loaded: `np` (numpy), `task`, and helper functions.

This workflow is iterative by design: each code block executes, you inspect output,
and then decide what to do next. Do not try to solve everything in one step.

Pre-loaded variables:
- `task["train"]`: list of {{"input": grid, "output": grid}} dicts
- `task["test"]`: list of {{"input": grid}} dicts (no output - you must predict)

Available helpers:
- `format_grid(grid)` - format grid as string for printing
- `accuracy(pred, expected)` - 1.0 if identical, 0.0 otherwise
- `soft_accuracy(pred, expected)` - fraction of matching cells

**Grid formatting:** Always use `format_grid()` when printing grids.

**Execution discipline:**
- Explore first: inspect data before transforming it (print samples, shapes/types, lengths, unique values).
- Iterate in small steps: write short snippets, inspect output, then refine; reuse persistent state.
- Verify before submitting: if outputs look empty/zero/unexpected (patterns, shapes, objects, colors, etc.), reassess your hypothesis and code.
- Submit only after inspection: `SUBMIT(...)` ends the run immediately, so print/check first and submit in a later step if needed.

---

**Response Format:**

[[ ## reasoning ## ]]
Your step-by-step analysis. Keep concise but clear.

[[ ## code ## ]]
```python
# Python code to execute
```

---

**Example:**

[[ ## reasoning ## ]]
Let me examine the training examples to understand the transformation pattern.

[[ ## code ## ]]
```python
print("Example 1 Input:")
print(format_grid(task["train"][0]["input"]))
print("\\nExample 1 Output:")
print(format_grid(task["train"][0]["output"]))
```

---

**Rules:**
- Verify your approach on training examples before submitting.
- Apply the rule to test inputs, inspect outputs, and iterate until the result is stable.
- Call `SUBMIT(test=[...])` only when you are confident in the final test predictions.
"""


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_arc_agi(
    data_folder: str,
    train_split: str = "training",
    eval_split: str = "evaluation",
) -> tuple[Dataset, Dataset]:
    """Load ARC-AGI tasks and format for SDPO training.

    Args:
        data_folder: Path to the arc-prize-2024 data directory.
        train_split: Name of the training split (default "training").
        eval_split: Name of the evaluation split (default "evaluation").

    Returns:
        (train_dataset, eval_dataset) with SDPO standard schema.
    """
    base = Path(data_folder)

    train_ds = _load_split(base, train_split)
    eval_ds = _load_split(base, eval_split)

    return train_ds, eval_ds


def _load_split(base: Path, split_name: str) -> Dataset:
    """Load a single ARC split into SDPO schema.

    Returns an empty Dataset if the challenge/solution files don't exist.
    """
    challenges_path = base / f"arc-agi_{split_name}_challenges.json"
    solutions_path = base / f"arc-agi_{split_name}_solutions.json"

    if not challenges_path.exists() or not solutions_path.exists():
        print(f"Warning: Missing {challenges_path} or {solutions_path}, returning empty dataset for split '{split_name}'")
        return Dataset.from_list([])

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)

    rows: list[dict] = []
    for task_id, task in challenges.items():
        train_pairs = task["train"]
        test_inputs = [t["input"] for t in task["test"]]
        test_outputs = solutions[task_id]

        question = format_task_question(train_pairs, test_inputs)

        # Ground truth: output grids + task data for the reward function
        tests_data = {
            "test": test_outputs,
            "train": train_pairs,
            "test_inputs": test_inputs,
        }

        rows.append({
            "kind": "arc_agi",
            "dataset": "arc_agi",
            "prompt": question,
            "answer": "-",
            "tests": json.dumps(tests_data),
            "description": f"ARC task {task_id}",
            "elo": "-",
            "system": SYSTEM_PROMPT.strip(),
            "task_id": task_id,
        })

    return Dataset.from_list(rows)
