"""ARC-AGI reward function with feedback for SDPO self-distillation.

Executes model-generated code in a subprocess interpreter, compares submitted
grids against ground truth, and produces rich feedback for teacher reprompting.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from verl.utils.reward_score.feedback.subprocess_interpreter import (
    CodeInterpreterError,
    FinalOutput,
    HistoryReset,
    SubprocessInterpreter,
)


# ---------------------------------------------------------------------------
# Response parsing (ported from rlvr envs/repl.py)
# ---------------------------------------------------------------------------

_SECTION_PATTERN = re.compile(
    r"\[\[\s*##\s*(\w+)\s*##\s*\]\]\s*\n(.*?)(?=\[\[\s*##|\Z)",
    re.DOTALL,
)

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_response(text: str) -> dict[str, str]:
    """Parse structured response into reasoning and code sections."""
    sections: dict[str, str] = {"reasoning": "", "code": ""}
    code_sections: list[str] = []

    matches = _SECTION_PATTERN.findall(text)
    for name, content in matches:
        name = name.lower().strip()
        if name == "code":
            code_sections.append(content.strip())
        elif name in sections:
            sections[name] = content.strip()

    if code_sections:
        all_code_blocks: list[str] = []
        for section_content in code_sections:
            code_matches = _CODE_BLOCK_PATTERN.findall(section_content)
            if code_matches:
                all_code_blocks.extend(m.strip() for m in code_matches)
            else:
                stripped = _strip_code_fences(section_content)
                if stripped:
                    all_code_blocks.append(stripped)
        sections["code"] = "\n".join(all_code_blocks)

    # Fallback: extract from markdown fences anywhere
    if not sections["code"]:
        code_matches = _CODE_BLOCK_PATTERN.findall(text)
        if code_matches:
            sections["code"] = "\n".join(m.strip() for m in code_matches)

    return sections


def _strip_code_fences(code: str | None) -> str:
    if not code:
        return ""
    result = code.strip()
    result = re.sub(r"^```(?:python|py)?\s*\n?", "", result)
    result = re.sub(r"\n?```\s*$", "", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Sandbox code (adapted from rlvr envs/repl.py)
# ---------------------------------------------------------------------------

ARC_SANDBOX_CODE = '''\
import json
import numpy as np

# --- ARC task data (auto-injected) ---
task = json.loads({task_json!r})

# --- Utility functions ---

def format_grid(grid):
    """Convert grid to string format."""
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()
    return "\\n".join(" ".join(str(cell) for cell in row) for row in grid)

def accuracy(pred, expected):
    """Compute exact match accuracy: 1.0 if grids are identical, 0.0 otherwise."""
    pred = np.array(pred)
    expected = np.array(expected)
    if pred.shape != expected.shape:
        return 0.0
    return 1.0 if np.array_equal(pred, expected) else 0.0

def soft_accuracy(pred, expected):
    """Compute cell-level accuracy: fraction of cells that match (0.0 to 1.0)."""
    pred = np.array(pred)
    expected = np.array(expected)
    if pred.shape != expected.shape:
        return 0.0
    return float(np.mean(pred == expected))
'''


# ---------------------------------------------------------------------------
# Grid comparison helpers
# ---------------------------------------------------------------------------

Grid = list[list[int]]


def _compute_exact_match(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0
    return 1.0 if all(g == e for g, e in zip(predicted, expected)) else 0.0


def _compute_cell_accuracy(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0
    total_cells = 0
    correct_cells = 0
    for pred, exp in zip(predicted, expected):
        pred_arr = np.array(pred)
        exp_arr = np.array(exp)
        if pred_arr.shape != exp_arr.shape:
            total_cells += exp_arr.size
            continue
        total_cells += exp_arr.size
        correct_cells += int(np.sum(pred_arr == exp_arr))
    return correct_cells / total_cells if total_cells > 0 else 0.0


def _compute_shape_match(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0
    matches = sum(
        1 for pred, exp in zip(predicted, expected)
        if np.array(pred).shape == np.array(exp).shape
    )
    return matches / len(expected)


def _compute_format_reward(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if len(predicted) == len(expected) else 0.0


def _convert_submit_data(data: dict[str, Any]) -> list[Grid] | str:
    """Convert SUBMIT data to list of grids. Returns error string on failure."""
    if "test" not in data:
        return "SUBMIT() must include test=... argument with predicted grids."
    grids = data["test"]
    if not isinstance(grids, list):
        return f"SUBMIT(test=...) must be a list of grids, got {type(grids).__name__}"
    converted = []
    for i, grid in enumerate(grids):
        if isinstance(grid, str):
            return f"SUBMIT(test[{i}]) is a string. Pass the numpy array or list directly."
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
            return f"SUBMIT(test[{i}]) must be a 2D list/array, got {type(grid).__name__}"
        try:
            converted.append([[int(c) for c in row] for row in grid])
        except (TypeError, ValueError) as e:
            return f"SUBMIT(test[{i}]) contains invalid cell values: {e}"
    return converted


# ---------------------------------------------------------------------------
# Feedback builder
# ---------------------------------------------------------------------------

def _format_arc_feedback(
    submitted: list[Grid] | None,
    expected: list[Grid],
    error_msg: str | None = None,
    no_submit: bool = False,
    submit_error: str | None = None,
) -> str:
    """Build rich feedback string for SDPO teacher reprompting."""
    if no_submit:
        return "No SUBMIT() call found. Your code must call SUBMIT(test=[...]) with your test predictions."

    if error_msg is not None:
        return f"Execution error:\n{error_msg}\n\nFix the error and try again."

    if submit_error is not None:
        return f"Invalid submission: {submit_error}\n\nFix the SUBMIT() call and try again."

    if submitted is None:
        return "No grids were submitted."

    parts = []
    all_correct = True

    for i, (pred, exp) in enumerate(zip(submitted, expected)):
        pred_arr = np.array(pred)
        exp_arr = np.array(exp)

        if pred == exp:
            parts.append(f"Test {i + 1}: Correct")
            continue

        all_correct = False
        exp_shape = f"{exp_arr.shape[0]}x{exp_arr.shape[1]}"
        pred_shape = f"{pred_arr.shape[0]}x{pred_arr.shape[1]}"

        lines = [f"Test {i + 1}: Wrong answer"]
        lines.append(f"  Expected shape: {exp_shape}, Got: {pred_shape}")

        if pred_arr.shape == exp_arr.shape:
            matching = int(np.sum(pred_arr == exp_arr))
            total = exp_arr.size
            lines.append(f"  Cell accuracy: {matching / total:.2f} ({matching}/{total} cells correct)")

            # Show first few mismatches
            mismatches = []
            for r in range(exp_arr.shape[0]):
                for c in range(exp_arr.shape[1]):
                    if pred_arr[r, c] != exp_arr[r, c]:
                        mismatches.append(f"({r},{c}): expected {exp_arr[r, c]}, got {pred_arr[r, c]}")
                    if len(mismatches) >= 5:
                        break
                if len(mismatches) >= 5:
                    break
            if mismatches:
                lines.append(f"  First mismatches: {'; '.join(mismatches)}")
        else:
            lines.append("  Shape mismatch — cannot compare cells.")

        parts.append("\n".join(lines))

    if all_correct:
        return "All test outputs correct."

    # Extra info about number wrong
    n_wrong = sum(1 for p, e in zip(submitted, expected) if p != e)
    header = f"{n_wrong}/{len(expected)} test case(s) wrong."

    return header + "\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Compute ARC-AGI reward by executing code and comparing grids.

    Args:
        solution_str: Model's full response text.
        ground_truth: JSON string with test output grids and task data.
        extra_info: Additional info dict (contains 'split', etc.)

    Returns:
        Dict with score, acc, pred, feedback, and diagnostic fields.
    """
    extra_info = extra_info or {}

    # Parse ground truth
    try:
        gt_data = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return _error_result("Failed to parse ground truth.")

    expected_outputs: list[Grid] = gt_data.get("test", [])
    if not expected_outputs:
        return _error_result("Ground truth missing 'test' output grids.")
    train_pairs = gt_data.get("train", [])
    test_inputs = gt_data.get("test_inputs", [])

    # Build task dict for sandbox
    task_data = {
        "train": train_pairs,
        "test": [{"input": inp} for inp in test_inputs],
    }
    task_json = json.dumps(task_data)

    # Parse model response
    sections = parse_response(solution_str)
    code = sections["code"]

    if not code:
        return _make_result(
            submitted=None,
            expected=expected_outputs,
            feedback="No code found in response. Provide code in [[ ## code ## ]] section.",
            no_submit=True,
        )

    # Execute in subprocess
    interpreter = SubprocessInterpreter(
        sandbox_code=ARC_SANDBOX_CODE.format(task_json=task_json),
        timeout=30.0,
    )

    try:
        interpreter.start()
        result = interpreter.execute(code)
    except CodeInterpreterError as e:
        error_msg = str(e)
        return _make_result(
            submitted=None,
            expected=expected_outputs,
            feedback=_format_arc_feedback(None, expected_outputs, error_msg=error_msg),
            error=True,
        )
    except SyntaxError as e:
        return _make_result(
            submitted=None,
            expected=expected_outputs,
            feedback=_format_arc_feedback(None, expected_outputs, error_msg=str(e)),
            error=True,
        )
    finally:
        interpreter.shutdown()

    # Handle FinalOutput (SUBMIT() was called) — execute() returns (FinalOutput, captured_output) tuple
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
        final_output, _captured = result
        converted = _convert_submit_data(final_output.output or {})
        if isinstance(converted, str):
            # Conversion error
            return _make_result(
                submitted=None,
                expected=expected_outputs,
                feedback=_format_arc_feedback(None, expected_outputs, submit_error=converted),
                no_submit=True,
            )
        return _make_result(
            submitted=converted,
            expected=expected_outputs,
            feedback=_format_arc_feedback(converted, expected_outputs),
        )

    # Handle HistoryReset (not expected in single-pass mode, but handle gracefully)
    if isinstance(result, HistoryReset):
        return _make_result(
            submitted=None,
            expected=expected_outputs,
            feedback="RESET_HISTORY() was called but no SUBMIT(). Call SUBMIT(test=[...]) to return predictions.",
            no_submit=True,
        )

    # No SUBMIT() called — code ran but didn't submit
    return _make_result(
        submitted=None,
        expected=expected_outputs,
        feedback=_format_arc_feedback(None, expected_outputs, no_submit=True),
        no_submit=True,
    )


def _make_result(
    submitted: list[Grid] | None,
    expected: list[Grid],
    feedback: str,
    error: bool = False,
    no_submit: bool = False,
) -> dict:
    """Build the standard result dict."""
    em = _compute_exact_match(submitted, expected)
    cell_acc = _compute_cell_accuracy(submitted, expected)
    shape = _compute_shape_match(submitted, expected)
    fmt = _compute_format_reward(submitted, expected)

    # Balanced reward: (2.0*em + 0.5*cell + 0.3*shape + 0.1*fmt) / 2.9
    score = (2.0 * em + 0.5 * cell_acc + 0.3 * shape + 0.1 * fmt) / 2.9

    return {
        "score": score,
        "acc": em,
        "pred": json.dumps(submitted) if submitted else "",
        "feedback": feedback,
        "cell_accuracy": cell_acc,
        "shape_match": shape,
        "no_submission": 1 if no_submit else 0,
        "error": 1 if error else 0,
    }


def _error_result(msg: str) -> dict:
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "feedback": msg,
        "cell_accuracy": 0.0,
        "shape_match": 0.0,
        "no_submission": 1,
        "error": 1,
    }
