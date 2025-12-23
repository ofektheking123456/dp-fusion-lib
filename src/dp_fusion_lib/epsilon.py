"""
Epsilon computation for differential privacy guarantees.

This module provides functions to compute (ε, δ)-DP guarantees from
per-step Rényi divergences, following the theory in:

    Thareja et al. "DP-Fusion: Token-Level Differentially Private
    Inference for Large Language Models" (arXiv:2507.04531)
"""

import math
from typing import Dict, List, Union


def compute_epsilon_single_group(
    divergences: List[float],
    alpha: float,
    delta: float,
    beta: float = None
) -> Dict[str, float]:
    """
    Compute (ε, δ)-DP guarantee for a single private group.

    For a single group (N=1), the per-step RDP formula simplifies to:
        eps_step = 4 * β_t

    where β_t = divergence_t / α (paper notation).

    Total epsilon:
        ε = (4/α) * Σ(divergences) + log(1/δ)/(α-1)

    Args:
        divergences: List of per-step D_α values (bounded by α·β internally).
        alpha: Rényi order (>1).
        delta: Target δ in (ε, δ)-DP.
        beta: Paper's β (where internal bound = α·β). If provided,
              also computes theoretical epsilon.

    Returns:
        Dict with:
            - "empirical": ε computed from actual divergences
            - "theoretical": ε assuming divergence = α·β at each step (if beta provided)
            - "T": number of tokens generated
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0,1)")

    T = len(divergences)
    log_delta_term = math.log(1.0 / delta) / (alpha - 1.0)

    # Empirical: divergences are bounded by α·β, so β_t = d/α
    # eps_t = 4 * β_t = 4 * (d / α)
    empirical_rdp = sum(4.0 * (d / alpha) for d in divergences)
    epsilon_empirical = empirical_rdp + log_delta_term

    result = {
        "empirical": epsilon_empirical,
        "T": T
    }

    # Theoretical: worst-case is divergence = α·β each step
    # β_t = β, so eps_t = 4 * β
    if beta is not None:
        theoretical_rdp = T * 4.0 * beta
        epsilon_theoretical = theoretical_rdp + log_delta_term
        result["theoretical"] = epsilon_theoretical

    return result


def compute_dp_epsilon(
    divergences: Dict[str, List[float]],
    alpha: float,
    delta: float,
    mode: str = "global"
) -> Union[float, Dict[str, float]]:
    """
    Compute (ε, δ)-DP guarantee from per-step Rényi divergences.

    Supports multi-group privacy with either global or per-group guarantees.

    Args:
        divergences: Mapping group_name -> list of β_t values (length=T).
                     The key "PUBLIC" (if present) will be ignored.
        alpha: Rényi order (>1).
        delta: Target δ in (ε, δ)-DP.
        mode: "global" for one ε protecting all groups (worst-case per step),
              "per_group" for a dict of ε_i per group.

    Returns:
        If mode == "global": float ε.
        If mode == "per_group": dict of {group: ε_i}.
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must be in (0,1)")

    # Filter out PUBLIC and ensure at least one private group
    priv_div = {g: lst for g, lst in divergences.items() if g != "PUBLIC"}
    if not priv_div:
        raise ValueError("No private groups provided")

    # Ensure all groups have same number of steps
    step_counts = {len(lst) for lst in priv_div.values()}
    if len(step_counts) != 1:
        raise ValueError(f"Divergence lists have unequal lengths: {step_counts}")

    T = step_counts.pop()
    N = len(priv_div)

    def eps_step(beta: float) -> float:
        """Compute per-step RDP cost."""
        if beta is None:
            raise ValueError("Found None in divergence list")
        arg = (N - 1.0) / N + (1.0 / N) * math.exp((alpha - 1.0) * 4.0 * beta)
        if arg <= 0.0:
            raise ValueError(f"Non-positive argument for log: {arg}")
        return (1.0 / (alpha - 1.0)) * math.log(arg)

    if mode == "global":
        total_rdp = 0.0
        for t in range(T):
            betas = [div_list[t] for div_list in priv_div.values()]
            beta_max = max(betas)
            total_rdp += eps_step(beta_max)
        epsilon = total_rdp + math.log(1.0 / delta) / (alpha - 1.0)
        return epsilon

    elif mode == "per_group":
        epsilons = {}
        for group, div_list in priv_div.items():
            total_rdp_g = 0.0
            for beta_t in div_list:
                total_rdp_g += eps_step(beta_t)
            eps_group = total_rdp_g + math.log(1.0 / delta) / (alpha - 1.0)
            epsilons[group] = eps_group
        return epsilons

    else:
        raise ValueError("mode must be 'global' or 'per_group'")
