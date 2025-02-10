"""
./backend/app/services/scoring.py

Contains the scoring function to calculate the final disease score based on normalized metrics.
"""

import math
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_score(
    *,
    # no default, better error than wrong value
    # handled by disease_service.py via config class
    coocc_norm: float,
    tfidf_norm: float,
    ratio_norm: float,
    pmi_norm: float,
    npmi_norm: float,
    llr_norm: float,
    base_weight: float,
    local_weight: float,
    global_weight: float,
    param_coocc: float,
    param_pmi: float,
    param_npmi: float,
    param_llr: float,
    mode: str,
) -> float:
    """
    Combines normalized metrics into a final disease score using various modes.

    Parameters
    ----------
    coocc_norm : float
        Normalized co-occurrence in [0,1].
    tfidf_norm : float
        Normalized TF-IDF in [0,1].
    ratio_norm : float
    pmi_norm : float
        Normalized PMI in [0,1].
    npmi_norm : float
        Normalized NPMI in [0,1].
    llr_norm : float
        Normalized LLR in [0,1].
    base_weight : float
        Weight for the base portion. (coocc + tfidf)
    local_weight : float
        Weight for the local ratio portion.
    global_weight : float
        Weight for the global portion. (pmi + npmi + llr)
    param_coocc : float
        Fraction for cooccurrence in base portion. (0..1)
    param_pmi : float
        Toggle for PMI. (0..1)
    param_npmi : float
        Toggle for NPMI. (0..1)
    param_llr : float
        Toggle for LLR. (0..1)
    mode : str
        "linear", "multiplicative_simple", "nonlinear_exponential", or "nonlinear_sigmoid"

    Returns
    -------
    float
        Final computed score.
    """
    block_sum = base_weight + local_weight + global_weight
    if abs(block_sum - 1.0) > 1e-7:
        logger.warning(f"[calculate_score] base/local/global weights sum={block_sum}, expected 1.0.")

    # 1) base portion
    base_coocc = param_coocc * coocc_norm
    base_tfidf = (1.0 - param_coocc) * tfidf_norm
    partial_base = base_coocc + base_tfidf

    # 2) local portion
    unscaled_part = partial_base * (1.0 - local_weight)
    scaled_part   = partial_base * local_weight * ratio_norm
    combined_base = unscaled_part + scaled_part

    # 3) global portion
    global_vals = []
    if param_pmi > 0:
        global_vals.append(pmi_norm)
    if param_npmi > 0:
        global_vals.append(npmi_norm)
    if param_llr > 0:
        global_vals.append(llr_norm)
    partial_global = sum(global_vals)/len(global_vals) if global_vals else 0.0

    logger.debug(
        f"[calculate_score] partial_base={partial_base:.3f}, "
        f"partial_local={combined_base:.3f}, partial_global={partial_global:.3f}, mode={mode}"
    )

    # 4) Combine partials based on mode
    if mode == "linear":
        # Standard linear blend
        score = (combined_base * (base_weight + local_weight)
                 + partial_global * global_weight)
        
    elif mode == "multiplicative_simple":
        # Multiply factors as (1 + portion)
        score = 1.0
        score *= (1.0 + (base_weight + local_weight) * combined_base)
        score *= (1.0 + global_weight * partial_global)
        score -= 1.0
    elif mode == "nonlinear_exponential":
        # Exponential approach with clamp to prevent extremely large values that could cause an overflow
        expo_val = ((base_weight + local_weight) * partial_base 
                    + global_weight * partial_global)
        expo_val = min(expo_val, 20.0)  # clamp to avoid OverflowError
        try:
            score = math.exp(expo_val) - 1.0
        except OverflowError:
            score = 1e10
    elif mode == "nonlinear_sigmoid":
        # Sigmoid transform
        linear_sum = ((base_weight + local_weight) * partial_base
                      + global_weight * partial_global)
        try:
            score = 1.0 / (1.0 + math.exp(-linear_sum))
        except OverflowError:
            score = 1.0 if linear_sum > 0 else 0.0
    else:
        raise ValueError(f"[calculate_score] Unknown mode: {mode}")

    logger.debug(f"[calculate_score] final score={score:.4f}")
    return score