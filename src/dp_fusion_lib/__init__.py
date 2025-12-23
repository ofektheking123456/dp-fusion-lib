"""
DP-Fusion-Lib: Token-Level Differentially Private Inference for LLMs

Generate text with formal (epsilon, delta)-differential privacy guarantees
using distribution fusion techniques.

This library implements the DP-Fusion algorithm from:

    Thareja et al. "DP-Fusion: Token-Level Differentially Private
    Inference for Large Language Models" (arXiv:2507.04531)

Quick Start:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from dp_fusion_lib import DPFusion, compute_epsilon_single_group
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    >>>
    >>> dpf = DPFusion(model=model, tokenizer=tokenizer)
    >>> dpf.add_message("system", "You are a helpful assistant.", is_private=False)
    >>> dpf.add_message("user", "My SSN is 123-45-6789. Summarize my info.", is_private=True)
    >>>
    >>> output = dpf.generate(alpha=2.0, beta=0.1, max_new_tokens=100)
    >>> print(output["text"])
    >>>
    >>> # Compute privacy guarantee
    >>> eps = compute_epsilon_single_group(
    ...     divergences=output["divergences"]["PRIVATE"],
    ...     alpha=2.0,
    ...     delta=1e-5,
    ...     beta=0.1
    ... )
    >>> print(f"Privacy: epsilon={eps['empirical']:.2f} at delta=1e-5")
"""

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required but not installed. Install it first:\n"
        "  pip install torch\n"
        "  or with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  or visit https://pytorch.org/get-started/locally/"
    ) from e

# Core classes and functions
from dp_fusion_lib.core import DPFusion, generate_dp_text
from dp_fusion_lib.tagger import Tagger, find_phrase_offsets
from dp_fusion_lib.epsilon import compute_epsilon_single_group, compute_dp_epsilon
from dp_fusion_lib._version import __version__

# Utility functions (advanced usage)
from dp_fusion_lib.utils import (
    compute_renyi_divergence_clipped_symmetric,
    find_lambda,
    replace_sequences_with_placeholder_fast,
    dp_fusion_groups_incremental,
    format_prompt_new_template,
    DEFAULT_BETA_DICT,
    ENTITY_TYPES,
    PLACEHOLDER_TOKEN,
)

__all__ = [
    # Main API
    "DPFusion",
    "Tagger",
    "generate_dp_text",
    # Epsilon computation
    "compute_epsilon_single_group",
    "compute_dp_epsilon",
    # Utility functions (advanced)
    "find_phrase_offsets",
    "compute_renyi_divergence_clipped_symmetric",
    "find_lambda",
    "replace_sequences_with_placeholder_fast",
    "dp_fusion_groups_incremental",
    "format_prompt_new_template",
    # Constants
    "DEFAULT_BETA_DICT",
    "ENTITY_TYPES",
    "PLACEHOLDER_TOKEN",
    # Version
    "__version__",
]
