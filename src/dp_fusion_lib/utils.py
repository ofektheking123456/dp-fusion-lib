"""
Utility functions for DP-Fusion.

This module contains the core algorithmic components:
- Rényi divergence computation
- Lambda search for privacy-utility tradeoff
- Token replacement for redaction
- Incremental DP-Fusion generation
"""

import math
from bisect import bisect_right
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


# Default beta values for different entity types
DEFAULT_BETA_DICT = {
    "PERSON": 0.5,
    "CODE": 0.5,
    "LOC": 0.5,
    "ORG": 0.5,
    "DEM": 0.5,
    "DATETIME": 0.5,
    "QUANTITY": 0.5,
    "MISC": 0.5,
}

# Entity types available
ENTITY_TYPES = [
    "PERSON", "CODE", "LOC", "ORG", "DEM",
    "DATETIME", "QUANTITY", "MISC"
]

# Default placeholder token for redaction
PLACEHOLDER_TOKEN = "_"


def replace_sequences_with_placeholder_fast(
    text: str,
    word_offsets: List[List[int]],
    placeholder: str,
    tokenizer
) -> List[int]:
    """
    Replace tokens falling within provided word offset ranges with placeholder tokens.

    Args:
        text: Original text string
        word_offsets: List of [start_char, end_char] offsets for words to replace
        placeholder: Placeholder token to use (e.g., "_")
        tokenizer: Tokenizer that returns 'input_ids' and 'offset_mapping'

    Returns:
        Token IDs with specified words replaced by placeholder token ID
    """
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']

    word_offsets = sorted(word_offsets, key=lambda x: x[0])
    starts = [wo[0] for wo in word_offsets]
    ends = [wo[1] for wo in word_offsets]

    for i, (t_start, t_end) in enumerate(offsets):
        if t_start == t_end:
            continue

        idx = bisect_right(starts, t_end)

        while idx > 0:
            idx -= 1
            w_start, w_end = starts[idx], ends[idx]

            if w_end > t_start and w_start < t_end:
                input_ids[i] = placeholder_id
                break

    return input_ids


def compute_renyi_divergence_clipped_symmetric(
    p: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute symmetric Rényi divergence D↔_α(p‖q) = max{D_α(p‖q), D_α(q‖p)}.

    Args:
        p: Probability vector (last dimension is the support)
        q: Probability vector (last dimension is the support)
        alpha: Rényi order (must be > 1)
        eps: Small constant for numerical stability

    Returns:
        D↔_α(p, q) with shape p.shape[:-1]
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")

    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)

    # Forward direction D_α(p‖q)
    term_pq = torch.sum(p.pow(alpha) * q.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_pq = (1.0 / (alpha - 1.0)) * torch.log(term_pq)

    # Reverse direction D_α(q‖p)
    term_qp = torch.sum(q.pow(alpha) * p.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_qp = (1.0 / (alpha - 1.0)) * torch.log(term_qp)

    return torch.maximum(div_pq, div_qp)


def find_lambda(
    p_priv: torch.Tensor,
    p_pub: torch.Tensor,
    alpha: float,
    beta: float,
    debug_mode: bool = False,
    max_iter: int = 20,
    tol: float = 1e-6
) -> Tuple[float, float]:
    """
    Binary search for λ in [0,1] that satisfies the divergence bound.

    Finds λ such that:
        D_α((1-λ)*p_pub + λ*p_priv || p_pub) <= beta

    Args:
        p_priv: Private distribution (already softmaxed & temperature-scaled)
        p_pub: Public distribution (already softmaxed & temperature-scaled)
        alpha: Rényi order (> 1)
        beta: Divergence threshold (>= 0)
        debug_mode: Whether to print debug information
        max_iter: Maximum binary search iterations
        tol: Tolerance for convergence

    Returns:
        Tuple of (lambda_value, divergence)
    """
    if beta <= 0:
        return 0.0, 0.0

    div_at_1 = compute_renyi_divergence_clipped_symmetric(p_priv, p_pub, alpha)

    if div_at_1 <= beta:
        return 1.0, div_at_1.item() if hasattr(div_at_1, 'item') else div_at_1

    left, right = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        mixture = mid * p_priv + (1 - mid) * p_pub
        div = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)

        if div > beta:
            right = mid
        else:
            left = mid

        if (right - left) < tol:
            break

    final_lambda = left
    mixture = final_lambda * p_priv + (1 - final_lambda) * p_pub
    final_div = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)

    return final_lambda, final_div.item() if hasattr(final_div, 'item') else final_div


def dp_fusion_groups_incremental(
    token_ids_groups: Dict[str, torch.Tensor],
    beta_dict: Dict[str, float],
    alpha: float,
    model,
    tokenizer,
    temperature: float = 1.0,
    max_new_tokens: int = 50,
    debug_mode: bool = False,
    device_map=None,
    batch_override=None
) -> Tuple[str, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    DP-Fusion generation with incremental decoding using KV-cache.

    Supports multi-group privacy where each group can have different β thresholds.

    Args:
        token_ids_groups: Dict mapping group names to token ID tensors.
                         Must include "PUBLIC" key for the redacted version.
        beta_dict: Mapping from group name to β threshold.
        alpha: Rényi divergence order (>1).
        model: HuggingFace CausalLM model.
        tokenizer: Corresponding tokenizer.
        temperature: Temperature for scaling logits.
        max_new_tokens: Maximum tokens to generate.
        debug_mode: Whether to print debug information.
        device_map: Optional device map.
        batch_override: Optional batch settings override.

    Returns:
        Tuple of (generated_text, lambdas_dict, divergences_dict)
    """
    eos_id = tokenizer.eos_token_id

    going_lambdas: Dict[str, List[float]] = {}
    going_divergence: Dict[str, List[float]] = {}

    if "PUBLIC" not in token_ids_groups:
        raise ValueError("Must have a 'PUBLIC' key in token_ids_groups.")

    private_groups = [g for g in token_ids_groups if g != "PUBLIC"]
    if not private_groups:
        raise ValueError("No private groups besides 'PUBLIC' – need at least one for DP-Fusion.")

    if device_map:
        first_device = next(iter(device_map.values()))
        device = torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    else:
        device = model.device

    for group, tokens in token_ids_groups.items():
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        token_ids_groups[group] = tokens.to(device)

    if debug_mode:
        print(f"[DP-Fusion] Starting generation. Private groups: {private_groups}")
        for g in token_ids_groups:
            print(f"[Initial] Prefix shape for group {g}: {token_ids_groups[g].shape}")

    group_order = list(token_ids_groups.keys())
    num_groups = len(group_order)

    # Initial pass: process each group's full prefix to build cache
    prefix_batches = [token_ids_groups[g] for g in group_order]
    input_batch = torch.nn.utils.rnn.pad_sequence(
        prefix_batches, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    if debug_mode:
        print(f"[Initial] Input batch shape: {input_batch.shape}")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        outputs = model(input_ids=input_batch, use_cache=True, past_key_values=None)

    past = outputs.past_key_values
    last_logits = outputs.logits[:, input_batch.size(1) - 1, :]
    group_logits = {g: last_logits[i] for i, g in enumerate(group_order)}

    pub_scaled = group_logits["PUBLIC"] / temperature
    p_pub = F.softmax(pub_scaled, dim=-1)

    p_priv_dict = {}
    for pg in private_groups:
        priv_scaled = group_logits[pg] / temperature
        p_priv_dict[pg] = F.softmax(priv_scaled, dim=-1)

    # DP-Fusion: find lambdas and form fused distribution
    lambdas = {}
    for pg in private_groups:
        beta_val = beta_dict.get(pg)
        lam_pg, got_div = find_lambda(p_priv_dict[pg], p_pub, alpha, beta_val, debug_mode=debug_mode)
        lambdas[pg] = lam_pg
        if debug_mode:
            print(f"[Initial] Selected Lambda for group {pg}: {lam_pg}, Divergence: {got_div}")

    sum_out = torch.zeros_like(p_pub)
    for pg in private_groups:
        lam_g = lambdas[pg]
        mix_g = lam_g * p_priv_dict[pg] + (1 - lam_g) * p_pub
        sum_out += mix_g
    p_out_avg = sum_out / len(private_groups)

    next_token = torch.multinomial(p_out_avg, 1).item()

    if debug_mode:
        token_str = tokenizer.decode([next_token])
        print(f"[Initial] Sampled token '{token_str}' (ID={next_token})")

    for g in group_order:
        token_ids_groups[g] = torch.cat(
            [token_ids_groups[g], torch.tensor([next_token], device=device)], dim=0
        )

    # Incremental loop
    for step in range(1, max_new_tokens):
        new_tokens_batch = torch.tensor([[next_token]] * num_groups, device=device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids=new_tokens_batch, past_key_values=past, use_cache=True)

        past = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]
        group_logits = {g: last_logits[i] for i, g in enumerate(group_order)}

        pub_scaled = group_logits["PUBLIC"] / temperature
        p_pub = F.softmax(pub_scaled, dim=-1)

        p_priv_dict = {}
        for pg in private_groups:
            priv_scaled = group_logits[pg] / temperature
            p_priv_dict[pg] = F.softmax(priv_scaled, dim=-1)

        lambdas = {}
        for pg in private_groups:
            beta_val = beta_dict.get(pg)
            lam_pg, div_got = find_lambda(p_priv_dict[pg], p_pub, alpha, beta_val, debug_mode=debug_mode)
            lambdas[pg] = lam_pg

            if debug_mode:
                print(f"[Step {step}] Selected Lambda for group {pg}: {lam_pg}, Divergence: {div_got}")

            if pg not in going_lambdas:
                going_lambdas[pg] = []
                going_divergence[pg] = []
            going_lambdas[pg].append(lam_pg)
            going_divergence[pg].append(div_got)

        sum_out = torch.zeros_like(p_pub)
        for pg in private_groups:
            mix_g = lambdas[pg] * p_priv_dict[pg] + (1 - lambdas[pg]) * p_pub
            sum_out += mix_g
        p_out_avg = sum_out / len(private_groups)

        next_token = torch.multinomial(p_out_avg, 1).item()

        for g in group_order:
            token_ids_groups[g] = torch.cat(
                [token_ids_groups[g], torch.tensor([next_token], device=device)], dim=0
            )

        del outputs, last_logits, group_logits
        torch.cuda.empty_cache()

        if next_token == eos_id:
            break

    final_text = tokenizer.decode(token_ids_groups["PUBLIC"], skip_special_tokens=True)

    if debug_mode:
        print("[DP-Fusion] Generation complete.")

    torch.cuda.empty_cache()

    return final_text, going_lambdas, going_divergence


def format_prompt_new_template(tokenizer, private_doc: str, placeholder: str) -> str:
    """
    Create a prompt using the HuggingFace chat template.

    Args:
        tokenizer: HuggingFace tokenizer
        private_doc: Document to paraphrase (may contain placeholders)
        placeholder: Placeholder character (e.g., "_")

    Returns:
        Formatted prompt for the model
    """
    system_text = (
        "You are given a passage that may contain placeholders (underscores) or incomplete data. "
        "Your job is to produce a natural paraphrase. "
        "Do not use any underscores or placeholders in your output. "
        "If data is missing, just omit it or paraphrase gracefully. "
        "Do not output anything except the paraphrase. "
        "Make sure retain all information from the source document."
    )

    user_task = (
        f"Document:\n\n{private_doc}\n\n"
        f"Paraphrase the above text. Whenever a placeholder i.e {placeholder} exists, you must completely ignore that information, "
        f"as {placeholder} indicates redacted text. To ensure the generated text is as natural as possible, "
        f"you must never output the {placeholder} themselves."
    )

    messages = [
        {"role": "user", "content": f"{system_text}\n\n{user_task}"},
        {"role": "assistant", "content": "Sure. Here is the paraphrased document without underscores or placeholders:"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return prompt
