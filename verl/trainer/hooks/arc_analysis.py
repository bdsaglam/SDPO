"""Pre-rollout hook that injects Gemini analysis into ARC-AGI prompts.

Called before rollout generation each epoch. Modifies the prompt messages
in-place and re-tokenizes to include analysis hints.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def inject_analysis(
    batch,
    config,
    tokenizer,
    cache: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """Inject Gemini analysis hints into ARC task prompts.

    Modifies the batch's raw_prompt messages and re-tokenizes input_ids
    for items that are arc_agi data sources.

    Args:
        batch: DataProto batch with non_tensor_batch containing raw_prompt,
               data_source, and extra_info/reward_model fields.
        config: Hydra config with arc_analysis settings.
        tokenizer: HuggingFace tokenizer for re-tokenization.
        cache: Dict mapping task_id -> analysis text. Updated in-place.

    Returns:
        (modified_batch, updated_cache)
    """
    if cache is None:
        cache = {}

    arc_analysis_cfg = config.get("arc_analysis", {})
    if not arc_analysis_cfg.get("enabled", False):
        return batch, cache

    # Find arc_agi items in the batch
    data_sources = batch.non_tensor_batch.get("data_source", np.array([]))
    if len(data_sources) == 0:
        return batch, cache

    arc_indices = [i for i, ds in enumerate(data_sources) if ds == "arc_agi"]
    if not arc_indices:
        return batch, cache

    # Lazy import to avoid requiring dspy when not used
    import dspy
    from data.analysis.arc_agi_analyzer import analyze_task

    # Initialize DSPy LM
    model_name = arc_analysis_cfg.get("model", "gemini/gemini-3-flash-preview")
    lm = dspy.LM(model_name)

    # Extract unique tasks that need analysis
    tasks_to_analyze: dict[str, dict] = {}
    task_id_per_index: dict[int, str] = {}

    for idx in arc_indices:
        # Get task data from reward_model ground_truth
        reward_model = batch.non_tensor_batch["reward_model"][idx]
        gt_str = reward_model.get("ground_truth", "")
        try:
            gt_data = json.loads(gt_str)
        except (json.JSONDecodeError, TypeError):
            continue

        # Derive task_id from description or index
        extra_info = batch.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, np.ndarray):
            extra_info = extra_info[idx] if idx < len(extra_info) else {}
        elif isinstance(extra_info, dict):
            pass
        else:
            extra_info = {}

        task_id = extra_info.get("index", str(idx))
        task_id_per_index[idx] = task_id

        if task_id not in cache and task_id not in tasks_to_analyze:
            # Build task dict for analysis
            train_pairs = gt_data.get("train", [])
            test_inputs = gt_data.get("test_inputs", [])
            tasks_to_analyze[task_id] = {
                "train": train_pairs,
                "test": [{"input": inp} for inp in test_inputs],
            }

    # Analyze uncached tasks
    if tasks_to_analyze:
        logger.info(f"Analyzing {len(tasks_to_analyze)} ARC tasks with {model_name}")
        for task_id, task_data in tasks_to_analyze.items():
            try:
                cache[task_id] = analyze_task(task_data, lm=lm)
                logger.info(f"  Analyzed task {task_id}: {len(cache[task_id])} chars")
            except Exception as e:
                logger.warning(f"  Analysis failed for task {task_id}: {e}")
                cache[task_id] = ""

    # Inject analysis into prompts and re-tokenize
    modified = False
    for idx in arc_indices:
        task_id = task_id_per_index.get(idx)
        if task_id is None or task_id not in cache or not cache[task_id]:
            continue

        analysis_text = cache[task_id]
        raw_prompt = batch.non_tensor_batch["raw_prompt"][idx]

        # Find the user message and prepend analysis hint
        for msg in raw_prompt:
            if msg["role"] == "user":
                hint_block = (
                    f"**Analysis Hint (transformation rule + strategy):**\n"
                    f"{analysis_text}\n\n"
                    f"---\n\n"
                )
                msg["content"] = hint_block + msg["content"]
                modified = True
                break

    return batch, cache
