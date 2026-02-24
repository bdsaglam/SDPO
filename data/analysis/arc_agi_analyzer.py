"""Gemini-based ARC task analyzer for SDPO hint injection.

Self-contained module that uses DSPy to analyze ARC tasks via Gemini,
producing transformation rule + strategy hints. Ported and simplified
from epiq's SingleShotAnalyzer.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import dspy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field descriptions (from epiq/analysis/prompts.py)
# ---------------------------------------------------------------------------

TRANSFORMATION_RULE_DESC = """Conceptual description of the transformation using objects, actions, and relationships.

Describe WHAT happens (extend, fill, copy, rotate) and WHERE/WHEN (until edge, toward marker, based on color).
Describe as a child would understand it — NO algorithms, NO indices, NO formulas.

**Use small ASCII diagrams to illustrate the concept** (not exact grids from examples):

Example for "extend lines until they hit a wall":
```
Before:    After:
. . X .    . . X .
. . . .    . . X .
. . . . →  . . X .
W W W W    W W W W
```

Keep diagrams small (3-5 rows/cols) showing the CONCEPT, not the actual puzzle grids."""

EXAMPLE_OBSERVATIONS_DESC = """Per-example observations. Keys: "example_A", "example_B", etc.

For each example: (1) input structure/pattern, (2) how the rule applies, (3) qualitative spatial info
(e.g., "small square in top-left", "L-shape near center", "marker in corner").

Do NOT give exact coordinates — the solver will determine those by inspecting the actual grid."""

STRATEGY_DESC = """Step-by-step application strategy using LOOK → DO → CHECK pattern.

Each step describes what to identify visually and what edit to make. Use Python for precision
when helpful (e.g., `grid[r:r2, c:c2] = color`), but don't write detection algorithms.

Format:
1. LOOK: [what to identify visually]
   DO: [what edit to make]
   CHECK: [how to verify]

2. LOOK: [next thing to identify]
   DO: [next edit]
   CHECK: [verify]

... continue until transformation is complete."""


# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


class AnalyzeWithStrategy(dspy.Signature):
    """Analyze ARC task examples and find the transformation rule.

    Study ALL examples carefully — both training examples (with answers) AND test inputs (without
    answers). Each example is labeled with a letter (A, B, C, ...).

    **1. Analyze Each Example Individually:**
      For each example, describe the SPECIFIC structure and pattern.
      Describe spatial relationships qualitatively — NOT exact coordinates.

    **2. Formulate the Transformation Rule:**
      The rule must explain ALL examples. Use ROLES not specifics:
      - Not "column 3" → "THE divider column"
      - Not "color 4" → "THE marker's color"

      Describe the rule as a child would understand it — not as an algorithm.

    **3. Provide an Application Strategy:**
      Step-by-step instructions using LOOK → DO → CHECK pattern.
      DO NOT describe algorithms. DO describe what to look for and what actions to take.

    If feedback is provided, use it to refine your analysis.
    """

    problem: str = dspy.InputField(
        desc="ARC task with labeled input-output examples. Grids are integers 0-9 representing colors."
    )
    feedback: str = dspy.InputField(
        desc="Feedback from previous failed attempt.",
        default="",
    )
    example_observations: dict[str, str] = dspy.OutputField(desc=EXAMPLE_OBSERVATIONS_DESC)
    transformation_rule: str = dspy.OutputField(desc=TRANSFORMATION_RULE_DESC)
    strategy: str = dspy.OutputField(desc=STRATEGY_DESC)


# ---------------------------------------------------------------------------
# Grid formatting for analysis
# ---------------------------------------------------------------------------


def format_grid(grid: list[list[int]]) -> str:
    """Format grid as comma-separated digits per row."""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def format_task_for_analysis(task: dict) -> str:
    """Format an ARC task dict for the analysis prompt.

    Uses letter labels (A, B, C, ...) for examples.
    """
    parts: list[str] = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    train_pairs = task.get("train", [])
    test_inputs = task.get("test", [])

    for i, pair in enumerate(train_pairs):
        label = letters[i] if i < len(letters) else f"Train{i}"
        parts.append(f"Example {label} (training):")
        parts.append("Input:")
        parts.append(format_grid(pair["input"]))
        parts.append("Output:")
        parts.append(format_grid(pair["output"]))
        parts.append("")

    for j, test in enumerate(test_inputs):
        idx = len(train_pairs) + j
        label = letters[idx] if idx < len(letters) else f"Test{j}"
        inp = test["input"] if isinstance(test, dict) else test
        parts.append(f"Example {label} (test — output unknown):")
        parts.append("Input:")
        parts.append(format_grid(inp))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_task(task: dict, lm: dspy.LM | None = None) -> str:
    """Analyze a single ARC task and return formatted hint string.

    Args:
        task: ARC task dict with 'train' and 'test' keys.
        lm: DSPy LM to use (defaults to configured LM).

    Returns:
        Formatted analysis string (transformation_rule + strategy).
    """
    problem = format_task_for_analysis(task)
    predictor = dspy.Predict(AnalyzeWithStrategy)

    ctx = dspy.context(lm=lm) if lm else nullcontext()
    with ctx:
        result = predictor(problem=problem, feedback="")

    analysis = str(result.transformation_rule)
    if result.strategy:
        analysis += f"\n\nApplication Strategy:\n{result.strategy}"
    return analysis


def analyze_batch(
    tasks: dict[str, dict],
    lm: dspy.LM | None = None,
) -> dict[str, str]:
    """Analyze multiple tasks, returns {task_id: analysis_text}.

    Sequential calls with error handling per task.
    """
    results: dict[str, str] = {}
    for task_id, task_data in tasks.items():
        try:
            results[task_id] = analyze_task(task_data, lm=lm)
        except Exception as e:
            logger.warning(f"Analysis failed for task {task_id}: {e}")
            results[task_id] = ""
    return results
