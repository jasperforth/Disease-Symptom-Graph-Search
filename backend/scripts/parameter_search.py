"""
parameter_search.py

Conducts a grid search over various parameter combinations to evaluate their effectiveness
in scoring disease associations based on symptoms. Compares each combination against a baseline
and records the detailed disease scores and rankings for each test case.

Usage:
    From the project root directory:
    python -m backend.scripts.parameter_search --top_k 5
    or
    python -m backend.scripts.parameter_search --top_k 3
"""

import itertools
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

from altair import param
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from backend.app.services.disease_service import DiseaseService
from backend.app.models.schemas import MultipleSymptomRequest, SymptomSeverity
from backend.app.core.config import settings, ROOT_DIR
from backend.app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Number of parallel jobs to run (set to -1 to use all available cores)
n_jobs = -1

###########################
# 1. PARAMETER GRIDS      #
###########################

# Weighted distribution ranges among base/local/global
BASE_WEIGHT_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]
LOCAL_WEIGHT_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]
GLOBAL_WEIGHT_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0]

# Toggles for global metrics
PARAM_COOC_RANGE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # fraction coocc in base
PARAM_PMI_RANGE = [0.0, 1.0]
PARAM_NPMI_RANGE = [0.0, 1.0]
PARAM_LLR_RANGE = [0.0, 1.0]

# Combination modes
MODE_OPTIONS = [
    "linear",
    "multiplicative_simple",
    "nonlinear_exponential",
    "nonlinear_sigmoid"
]

###########################
# 2. BASELINE PARAMETERS  #
###########################
BASELINE_PARAMS = {
    'mode': "linear",            
    'base_weight': 1.0,
    'local_weight': 0.0,
    'global_weight': 0.0,
    'param_coocc': 0.5,
    'param_pmi': 0.0,
    'param_npmi': 0.0,
    'param_llr': 0.0
}

###########################
# 3. HELPER FUNCTIONS     #
###########################
def _normalize_weights(b: float, l: float, g: float) -> Tuple[float, float, float]:
    """
    Normalize b, l, g to sum to 1.0. If they are all zero, return (1.0, 0.0, 0.0).
    """
    s = b + l + g
    if abs(s) < 1e-9:
        # All zero => default to base=1.0
        return (1.0, 0.0, 0.0)
    return (b / s, l / s, g / s)

def generate_test_cases() -> List[MultipleSymptomRequest]:
    """
    Generate a list of test cases (MultipleSymptomRequest objects).
    Each test case includes multiple symptoms with severities.
    """
    test_cases = [

        # Case 1: 1 symptom (common)
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Fever", severity="Medium")
        ]),

        # Case 2: 1 symptom 
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Headache", severity="Medium")
        ]),

        # Case 3: 1 symptoms
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Korsakoff Syndrome", severity="Medium"),
        ]),

        # Case 4: 2 symptoms
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Korsakoff Syndrome", severity="Medium"),
            SymptomSeverity(name="Memory Disordersr", severity="Medium"),
        ]),

        # Case 5: 3 symptoms
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Headache", severity="Medium"),
            SymptomSeverity(name="Fever", severity="Medium")
        ]),

        # Case 6: 3 symptoms
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Headache", severity="Medium"),
            SymptomSeverity(name="Fever", severity="Medium"),
            SymptomSeverity(name="Vomiting", severity="Medium")
        ]),
    
        # Case 7: 8 symptoms
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Headache", severity="Medium"),
            SymptomSeverity(name="Fever", severity="Medium"),
            SymptomSeverity(name="Vision Disorders", severity="Medium"),
            SymptomSeverity(name="Vomiting", severity="Medium")
        ]),

        # Case 8: 1 symptom (common disease: hypertension, low cooccurrence)
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Flatulence", severity="Medium")
        ]),

        # Case 9: 1 symptom (rare disease with degree > 5: alstrom syndrome, low cooccurrence)
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Hearing Loss", severity="Medium")
        ]),

        # Case 10: 1 symptom (common disease: hypertension, high cooccurrence)
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Proteinuria", severity="Medium")
        ]),

        # Case 11: 1 symptom (rare disease degree > 5: alstrom syndrome, high ccooccurrence)
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Obesity", severity="Medium")
        ]),

        # Case 12: 3 symptom (combined cases 11, 13, + low cooccurrence )
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Flatulence", severity="Medium"),
            SymptomSeverity(name="Proteinuria", severity="Medium"),
            SymptomSeverity(name="Prostatism", severity="Medium")
        ]),

        # Case 13: 2 symptom (combined cases 12, 14, plus  low cooccurrence )
        MultipleSymptomRequest(symptoms=[
            SymptomSeverity(name="Hearing Loss", severity="Medium"),
            SymptomSeverity(name="Obesity", severity="Medium"),
            SymptomSeverity(name="Arthralgia", severity="Medium"),

        ]),
    ]

    return test_cases

def define_parameter_grid() -> Dict[str, List]:
    """
    Define the parameter grid for experimentation.
    We only include the parameters your disease_service.py expects:
        - mode
        - base_weight, local_weight, global_weight
        - param_coocc, param_pmi, param_npmi, param_llr
    """
    param_grid = {
        # Combination mode
        'mode': MODE_OPTIONS,

        # Weighted distribution among base/local/global
        'base_weight':   BASE_WEIGHT_RANGE,
        'local_weight':  LOCAL_WEIGHT_RANGE,
        'global_weight': GLOBAL_WEIGHT_RANGE,

        # Toggles for global metrics (0 => off, 1 => on)
        'param_coocc': PARAM_COOC_RANGE,  # fraction coocc in base
        'param_pmi':   PARAM_PMI_RANGE,
        'param_npmi':  PARAM_NPMI_RANGE,
        'param_llr':   PARAM_LLR_RANGE
    }
    return param_grid

def get_top_diseases(
    disease_service: DiseaseService, 
    test_case: MultipleSymptomRequest, 
    params: dict, 
    top_k: int = 10
) -> List:
    """
    Compute the top diseases for a given test case and parameter settings.
    Relying on disease_service logic to re-normalize weights as needed.
    """
    diseases = disease_service.get_diseases_by_symptom(
        payload=test_case,
        mode=params['mode'],

        base_weight=params['base_weight'],
        local_weight=params['local_weight'],
        global_weight=params['global_weight'],

        param_coocc=params['param_coocc'],
        param_pmi=params['param_pmi'],
        param_npmi=params['param_npmi'],
        param_llr=params['param_llr']
    )
    return diseases[:top_k]

def compute_overlap(baseline_diseases: List, combo_diseases: List) -> float:
    """
    Compute the overlap in disease names between two top-k lists.
    """
    baseline_names = set(d.disease.lower() for d in baseline_diseases)
    combo_names = set(d.disease.lower() for d in combo_diseases)
    if not baseline_names:
        return 0.0
    intersection = baseline_names.intersection(combo_names)
    return len(intersection) / len(baseline_names)

def evaluate_combination(
    combo_params: Dict, 
    test_cases: List[MultipleSymptomRequest],
    baseline_results: List[List],
    top_k: int
) -> Dict:
    """
    Evaluate a single parameter combination against all test cases.
    Runs in parallel via Joblib.
    
    Returns a dictionary containing:
        - All parameter settings
        - average_score: Average of summed scores across test cases
        - average_overlap_with_baseline: Average overlap ratio with baseline across test cases
        - detailed_results: JSON string containing detailed disease-score pairs per test case
    """
    from backend.app.services.disease_service import DiseaseService

    try:
        # Each worker gets its own DiseaseService instance
        disease_service = DiseaseService()
    except Exception as e:
        logger.error(f"Error initializing DiseaseService in worker: {e}")
        return None

    total_score_sum = 0.0
    overlap_sum = 0.0
    detailed_results = []

    for test_case, baseline_top in zip(test_cases, baseline_results):
        try:
            # Retrieve top diseases using the current parameter combination
            combo_top = get_top_diseases(disease_service, test_case, combo_params, top_k=top_k)
            # Summation of scores for this test case
            test_case_score = sum(d.score for d in combo_top)
            total_score_sum += test_case_score

            # Overlap with baseline for this test case
            overlap = compute_overlap(baseline_top, combo_top)
            overlap_sum += overlap

            # Store detailed disease-score pairs
            detailed_results.append({
                'test_case': {
                    'symptoms': [(sym.name, sym.severity) for sym in test_case.symptoms]
                },
                'top_diseases': [{'disease': d.disease, 'score': d.score} for d in combo_top]
            })

        except Exception as e:
            logger.error(f"Error evaluating combination {combo_params}: {e}")
            continue  # skip this test case

    n_cases = len(test_cases)
    avg_score = total_score_sum / n_cases if n_cases else 0.0
    avg_overlap = overlap_sum / n_cases if n_cases else 0.0

    # Merge results back
    result = combo_params.copy()
    result['average_score'] = avg_score
    result['average_overlap_with_baseline'] = avg_overlap
    # Convert the list of results to JSON so we can store them in the CSV
    result['detailed_results'] = json.dumps(detailed_results)  
    return result

def compute_llr(k11, k12, k21, k22):
    """
    Compute the Log-Likelihood Ratio (LLR) for given contingency table values.
    """
    from scipy.stats import chi2_contingency

    table = [[k11, k12], [k21, k22]]
    try:
        chi2, p, dof, ex = chi2_contingency(table, correction=False)
        return chi2
    except Exception as e:
        logger.error(f"Error computing LLR with table {table}: {e}")
        return 0.0

def minmax_norm(value, min_val, max_val):
    """
    Apply min-max normalization to a value.
    """
    if max_val - min_val == 0:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def calculate_score(
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
    mode: str
) -> float:
    """
    Calculate the final score based on normalized metrics and weights.
    """
    base_score = (param_coocc * coocc_norm) + (1 - param_coocc) * tfidf_norm
    local_score = ratio_norm 
    global_score = (param_pmi * pmi_norm) + (param_npmi * npmi_norm) + (param_llr * llr_norm)

    # Combine scores based on mode
    if mode == "linear":
        final_score = (base_weight * base_score) + (local_weight * local_score) + (global_weight * global_score)
    elif mode == "multiplicative_simple":
        final_score = (base_weight * base_score) * (local_weight * local_score + global_weight * global_score)
    elif mode == "nonlinear_exponential":
        final_score = (base_weight * (base_score ** 2)) + (local_weight * (local_score ** 2)) + (global_weight * (global_score ** 2))
    elif mode == "nonlinear_sigmoid":
        import math
        final_score = (base_weight * (1 / (1 + math.exp(-base_score)))) + \
                      (local_weight * (1 / (1 + math.exp(-local_score)))) + \
                      (global_weight * (1 / (1 + math.exp(-global_score))))
    else:
        final_score = 0.0  # Default case

    return final_score

###########################
# 4. MAIN ENTRY POINT     #
###########################
def main():
    """
    Main function to perform parameter grid search.
    Evaluates each parameter combination against all test cases and logs the results.
    Allows specifying top_k via command-line arguments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Parameter Search for Disease Scoring")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top diseases to retrieve per test case (e.g., 3 or 10)")
    args = parser.parse_args()
    top_k = args.top_k

    logger.info(f"Starting parameter search with top_k={top_k}.")
    print(f"Starting parameter search with top_k={top_k}.")

    # 1) Initialize DiseaseService for baseline
    try:
        disease_service = DiseaseService()
        print("Initialized DiseaseService (main thread).")
        logger.info("Initialized DiseaseService (main thread).")
    except Exception as e:
        print(f"Error initializing DiseaseService: {e}")
        logger.error(f"Error initializing DiseaseService: {e}")
        sys.exit(1)

    # 2) Generate test cases
    test_cases = generate_test_cases()
    print(f"Generated {len(test_cases)} test cases.")
    logger.info(f"Generated {len(test_cases)} test cases.")

    # 3) Define parameter grid
    param_grid = define_parameter_grid()
    keys, values = zip(*param_grid.items())
    raw_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info(f"Raw parameter combinations: {len(raw_param_combinations)}")
    print(f"Raw parameter combinations: {len(raw_param_combinations)}")

    # 4) Deduplicate parameter combinations
    visited = set()
    deduped_param_combinations = []

    for combo in raw_param_combinations:
        b = combo['base_weight']
        l = combo['local_weight']
        g = combo['global_weight']

        # If all global metrics are zero, force global_weight=0 and renormalize base and local
        if combo['param_pmi'] == 0.0 and combo['param_npmi'] == 0.0 and combo['param_llr'] == 0.0:
            b_norm, l_norm, _ = _normalize_weights(b, l, 0.0)
            g_norm = 0.0
        else:
            b_norm, l_norm, g_norm = _normalize_weights(b, l, g)

        # Round to avoid floating point precision issues
        final_tuple = (
            round(b_norm, 4),
            round(l_norm, 4),
            round(g_norm, 4),
            round(combo['param_coocc'], 4),
            round(combo['param_pmi'], 4),
            round(combo['param_npmi'], 4),
            round(combo['param_llr'], 4),
            combo['mode']
        )

        if final_tuple not in visited:
            visited.add(final_tuple)
            # Keep raw values
            new_combo = combo.copy()
            new_combo['raw_base_weight'] = combo['base_weight']
            new_combo['raw_local_weight'] = combo['local_weight']
            new_combo['raw_global_weight'] = combo['global_weight']
            # Replace with normalized values for scoring/results
            new_combo['base_weight'] = b_norm
            new_combo['local_weight'] = l_norm
            new_combo['global_weight'] = g_norm
            deduped_param_combinations.append(new_combo)

    param_combinations = deduped_param_combinations
    # param_combinations = raw_param_combinations
    total_combinations = len(param_combinations)
    logger.info(f"Deduplicated parameter combinations: {total_combinations}")
    print(f"Deduplicated parameter combinations: {total_combinations}")

    # 5) Prepare CSV file for results
    from pathlib import Path

    data_dir = Path(ROOT_DIR)/ "data" / "experiments_param_search"
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the data directory exists
    csv_path = data_dir / f"parameter_search_comparison_topk_{top_k}.csv"
    # We'll store the param columns + aggregated metrics + 'detailed_results'
    df_columns = list(param_grid.keys()) + [
        'average_score', 'average_overlap_with_baseline', 'detailed_results'
    ]

    # 6) Compute baseline results
    print("Computing baseline results...")
    logger.info("Computing baseline results...")
    baseline_results = []
    for i, test_case in enumerate(test_cases, start=1):
        try:
            baseline_top = get_top_diseases(disease_service, test_case, BASELINE_PARAMS, top_k=top_k)
            baseline_results.append(baseline_top)
            logger.debug(f"Baseline Test Case {i}: {[d.disease for d in baseline_top]}")
        except Exception as e:
            logger.error(f"Error computing baseline for test case {i}: {e}")
            baseline_results.append([])  # Insert empty or handle as needed
    print("Baseline computation completed.")
    logger.info("Baseline computation completed.")

    # 7) Evaluate each parameter combo vs. baseline
    print("Starting parameter evaluations...")
    logger.info("Starting parameter evaluations...")

    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_combination)(
                combo_params=combo_params,
                test_cases=test_cases,
                baseline_results=baseline_results,
                top_k=top_k
            )
            for combo_params in tqdm(param_combinations, desc="Evaluating Combos")
        )
    except Exception as e:
        print(f"Error during parallel execution: {e}")
        logger.error(f"Error during parallel execution: {e}")
        sys.exit(1)

    # Filter out None results
    results = [res for res in results if res is not None]

    # 8) Create DataFrame and save
    if results:
        results_df = pd.DataFrame(results, columns=df_columns)
        try:
            results_df.to_csv(csv_path, index=False)
            print(f"Done. Detailed results stored in '{csv_path}'.")
            logger.info(f"Parameter search completed. Results saved to '{csv_path}'.")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            logger.error(f"Error saving results to CSV: {e}")
            sys.exit(1)
    else:
        print("No results to save.")
        logger.warning("No results to save.")

if __name__ == "__main__":
    main()