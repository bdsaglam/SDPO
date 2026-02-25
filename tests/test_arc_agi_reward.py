"""Smoke test for ARC-AGI reward function — run before training to catch silent failures.

Usage: python tests/test_arc_agi_reward.py
"""

import json
import importlib
import importlib.util
import sys
import traceback

sys.path.insert(0, ".")

# Direct import to avoid heavy verl.__init__ (needs packaging, torch, etc.)
_spec = importlib.util.spec_from_file_location(
    "arc_agi", "verl/utils/reward_score/feedback/arc_agi.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_score = _mod.compute_score


# Minimal ARC task: 1x1 grid, input=[[0]], output=[[1]]
GROUND_TRUTH = json.dumps({
    "test": [[[1]]],             # expected output grids
    "train": [{"input": [[0]], "output": [[1]]}],
    "test_inputs": [[[0]]],
})


def test_correct_submission():
    """Model generates correct code that calls SUBMIT()."""
    response = """
[[ ## reasoning ## ]]
The pattern is: replace 0 with 1.

[[ ## code ## ]]
```python
output = [[1]]
SUBMIT(test=[output])
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["acc"] == 1.0, f"Expected acc=1.0, got {result['acc']}"
    assert result["error"] == 0, f"Unexpected error: {result.get('feedback')}"
    assert result["no_submission"] == 0, "Should have detected SUBMIT()"
    assert result["score"] > 0.9, f"Expected high score, got {result['score']}"
    print(f"  PASS: correct submission -> score={result['score']:.3f}, acc={result['acc']}")


def test_wrong_submission():
    """Model submits wrong grid."""
    response = """
[[ ## code ## ]]
```python
SUBMIT(test=[[[9]]])
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["acc"] == 0.0, f"Expected acc=0.0, got {result['acc']}"
    assert result["error"] == 0, "Wrong answer shouldn't be an error"
    assert result["score"] < 1.0, f"Wrong answer should have low score, got {result['score']}"
    print(f"  PASS: wrong submission -> score={result['score']:.3f}, acc={result['acc']}")


def test_no_submit_call():
    """Model runs code but forgets SUBMIT()."""
    response = """
[[ ## code ## ]]
```python
x = 42
print(x)
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["no_submission"] == 1, "Should detect missing SUBMIT()"
    assert result["score"] == 0.0, f"No submission should score 0, got {result['score']}"
    print(f"  PASS: no SUBMIT() -> score={result['score']:.3f}, no_submission={result['no_submission']}")


def test_syntax_error():
    """Model generates code with syntax error."""
    response = """
[[ ## code ## ]]
```python
def broken(
    # missing closing paren and body
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["error"] == 1 or result["no_submission"] == 1, \
        f"Syntax error should flag error or no_submission, got {result}"
    assert result["score"] == 0.0, f"Error should score 0, got {result['score']}"
    print(f"  PASS: syntax error -> score={result['score']:.3f}, error={result['error']}")


def test_runtime_error():
    """Model generates code that crashes at runtime."""
    response = """
[[ ## code ## ]]
```python
x = 1 / 0
SUBMIT(test=[[[1]]])
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["error"] == 1, f"Runtime error should flag error=1, got {result}"
    assert result["score"] == 0.0, f"Error should score 0, got {result['score']}"
    print(f"  PASS: runtime error -> score={result['score']:.3f}, error={result['error']}")


def test_no_code_in_response():
    """Model responds with text but no code."""
    response = "I think the answer is 1 but I'm not going to write code."
    result = compute_score(response, GROUND_TRUTH)
    assert result["no_submission"] == 1, f"No code should flag no_submission, got {result}"
    print(f"  PASS: no code -> score={result['score']:.3f}, no_submission={result['no_submission']}")


def test_sandbox_has_task_variable():
    """Verify the sandbox correctly injects the `task` variable."""
    response = """
[[ ## code ## ]]
```python
# The sandbox should have `task` pre-loaded
assert 'train' in task, f"task missing 'train': {task.keys()}"
assert 'test' in task, f"task missing 'test': {task.keys()}"
output = task['train'][0]['output']
SUBMIT(test=[output])
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert result["error"] == 0, f"Sandbox task variable broken: {result.get('feedback')}"
    assert result["acc"] == 1.0, f"Expected acc=1.0 using task variable, got {result['acc']}"
    print(f"  PASS: sandbox task variable -> score={result['score']:.3f}, acc={result['acc']}")


def test_feedback_field_populated():
    """Verify feedback is always a non-empty string (needed for SDPO reprompting)."""
    # Test with error
    result = compute_score("no code here", GROUND_TRUTH)
    assert isinstance(result.get("feedback"), str) and len(result["feedback"]) > 0, \
        f"Feedback should be non-empty string, got: {result.get('feedback')!r}"

    # Test with correct answer
    response = """
[[ ## code ## ]]
```python
SUBMIT(test=[[[1]]])
```
"""
    result = compute_score(response, GROUND_TRUTH)
    assert isinstance(result.get("feedback"), str) and len(result["feedback"]) > 0, \
        f"Feedback should be non-empty even for correct answers, got: {result.get('feedback')!r}"
    print(f"  PASS: feedback always populated")


if __name__ == "__main__":
    tests = [
        test_correct_submission,
        test_wrong_submission,
        test_no_submit_call,
        test_syntax_error,
        test_runtime_error,
        test_no_code_in_response,
        test_sandbox_has_task_variable,
        test_feedback_field_populated,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL: {test.__name__}")
            traceback.print_exc()
            print()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    else:
        print("All reward function tests passed — safe to launch training.")
