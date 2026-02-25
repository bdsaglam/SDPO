"""ARC-AGI multi-turn interaction for SDPO training.

Executes model-generated code in a persistent subprocess REPL, returning
stdout/stderr as the next user message.  When SUBMIT() is called, scores
the submission and terminates the interaction.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction
from verl.utils.reward_score.feedback.arc_agi import (
    ARC_SANDBOX_CODE,
    _compute_cell_accuracy,
    _compute_exact_match,
    _compute_format_reward,
    _compute_shape_match,
    _convert_submit_data,
    _format_arc_feedback,
    parse_response,
)
from verl.utils.reward_score.feedback.subprocess_interpreter import (
    CodeInterpreterError,
    FinalOutput,
    HistoryReset,
    SubprocessInterpreter,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ArcAgiInteraction(BaseInteraction):
    """Multi-turn ARC-AGI interaction.

    Each turn:
      1. Parse ``[[ ## code ## ]]`` from the assistant message.
      2. Execute in a persistent ``SubprocessInterpreter`` (state carries over).
      3. If ``SUBMIT()`` → score and terminate.
      4. Otherwise → return stdout/stderr, continue.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # Parse ground truth
        gt_data = json.loads(ground_truth)
        expected_outputs = gt_data.get("test", [])
        train_pairs = gt_data.get("train", [])
        test_inputs = gt_data.get("test_inputs", [])

        # Build task dict for sandbox injection
        task_data = {
            "train": train_pairs,
            "test": [{"input": inp} for inp in test_inputs],
        }
        task_json = json.dumps(task_data)

        # Create and start a persistent interpreter
        interpreter = SubprocessInterpreter(
            sandbox_code=ARC_SANDBOX_CODE.format(task_json=task_json),
            timeout=30.0,
        )
        interpreter.start()

        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "expected_outputs": expected_outputs,
            "interpreter": interpreter,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        inst = self._instance_dict[instance_id]
        interpreter: SubprocessInterpreter = inst["interpreter"]
        expected: list = inst["expected_outputs"]

        # Extract latest assistant message
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                break

        # Parse code from response
        sections = parse_response(content)
        code = sections["code"]

        if not code:
            return (
                False,
                "No code block found. Provide code in a [[ ## code ## ]] section.",
                0.0,
                {},
            )

        # Execute in persistent REPL
        try:
            result = interpreter.execute(code)
        except CodeInterpreterError as e:
            return False, f"Execution error:\n{e}\n\nFix the error and try again.", 0.0, {}
        except SyntaxError as e:
            return False, f"Syntax error:\n{e}\n\nFix the error and try again.", 0.0, {}

        # SUBMIT() was called
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
            final_output, captured = result
            converted = _convert_submit_data(final_output.output or {})

            if isinstance(converted, str):
                # Conversion error — let model retry
                feedback = _format_arc_feedback(None, expected, submit_error=converted)
                return False, feedback, 0.0, {}

            # Score the submission
            em = _compute_exact_match(converted, expected)
            cell_acc = _compute_cell_accuracy(converted, expected)
            shape = _compute_shape_match(converted, expected)
            fmt = _compute_format_reward(converted, expected)
            score = (2.0 * em + 0.5 * cell_acc + 0.3 * shape + 0.1 * fmt) / 2.9

            feedback = _format_arc_feedback(converted, expected)
            inst["reward"] = score

            return True, feedback, score, {
                "acc": em,
                "cell_accuracy": cell_acc,
                "shape_match": shape,
            }

        # RESET_HISTORY
        if isinstance(result, HistoryReset):
            return False, f"History reset: {result.summary}", 0.0, {}

        # Normal output — return stdout to model
        stdout = result if isinstance(result, str) else str(result)
        if not stdout.strip():
            stdout = "(Code executed successfully with no output.)"
        return False, stdout, 0.0, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict.get(instance_id, {}).get("reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        inst = self._instance_dict.pop(instance_id, None)
        if inst and "interpreter" in inst:
            try:
                inst["interpreter"].shutdown()
            except Exception:
                pass
