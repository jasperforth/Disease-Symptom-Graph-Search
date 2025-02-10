"""
compute_metric_bounds.py

Queries ALL edges from Neo4j, computes min/max for each metric (including LLR),
and stores them in data/preprocessed/metric_bounds.json (or CSV) for later use.

TODO Store them in the Meta node?
"""

import json
import math
from pathlib import Path

from backend.app.database.connection import neo4j_conn
from backend.app.core.config import ROOT_DIR

def compute_llr(k11: float, k12: float, k21: float, k22: float) -> float:
    """
    Computes the Log-Likelihood Ratio (LLR) for a contingency table.

    Parameters
    ----------
    k11 : float
        Number of co-occurrences (disease and symptom present).
    k12 : float
        Number of publications with symptom but without disease.
    k21 : float
        Number of publications with disease but without symptom.
    k22 : float
        Number of publications without disease and without symptom.

    Returns
    -------
    float
        The computed LLR value. Returns 0.0 in case of computation error.
    """
    epsilon = 1e-10
    k11 += epsilon
    k12 += epsilon
    k21 += epsilon
    k22 += epsilon

    total = k11 + k12 + k21 + k22
    E11 = ((k11 + k12) * (k11 + k21)) / total
    E12 = (k11 + k12) - E11
    E21 = (k11 + k21) - E11
    E22 = total - (E11 + E12 + E21)

    try:
        llr_val = 2 * (
            k11 * math.log(k11 / E11) +
            k12 * math.log(k12 / E12) +
            k21 * math.log(k21 / E21) +
            k22 * math.log(k22 / E22)
        )
    except (ValueError, OverflowError):
        llr_val = 0.0

    return llr_val

def trimmed_minmax(values: list) -> tuple:
    """
    Computes the trimmed min and max by ignoring the top and bottom 1% of values.

    Parameters
    ----------
    values : list
        List of numerical values.

    Returns
    -------
    tuple
        (min_value, max_value) after outlier trimming.
        Returns (0.0, 1.0) if the list is empty.
    """
    if not values:
        return (0.0, 1.0)
    vsorted = sorted(values)
    n = len(vsorted)
    lower_i = max(0, int(0.01 * n))
    upper_i = min(n - 1, int(0.99 * n))
    sub = vsorted[lower_i:upper_i + 1]
    return (min(sub), max(sub))

def min_max(values: list) -> tuple:
    """
    Computes the min and max values from a list of numerical values.

    Parameters
    ----------
    values : list
        List of numerical values.

    Returns
    -------
    tuple
        (min_value, max_value) from the list.
        Returns (0.0, 1.0) if the list is empty.
    """
    if not values:
        return (0.0, 1.0)
    return (min(values), max(values))

def main():
    """
    Main function to compute min and max bounds for each metric (including LLR)
    and store them in data/metric_bounds.json.
    """
    # Step 1: Retrieve total_coocc from the Meta node
    db = neo4j_conn
    meta_query = """
    MATCH (m:Meta {name: 'GlobalStats'})
    RETURN m.total_cooccurrence AS total_coocc
    """
    meta_res = db.execute_query(meta_query, {})
    if not meta_res:
        print("Meta node with total_cooccurrence not found. Aborting.")
        return
    N_ds = meta_res[0]["total_coocc"]

    # Step 2: Retrieve all edges with necessary properties
    query = """
    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
    RETURN
      r.cooccurrence AS coocc,
      r.tfidf_score AS tfidf,
      r.PMI AS pmi,
      r.NPMI AS npmi,
      s.pubmed_occurrence AS s_occ,
      d.pubmed_occurrence AS d_occ,
      r.ratio AS ratio
    """
    results = db.execute_query(query, {})

    # Step 3: Accumulate metrics and compute LLR
    coocc_list, tfidf_list = [], []
    pmi_list, npmi_list = [], []
    llr_list, ratio_list = [], []

    for r in results:
        coocc = r["coocc"] if r["coocc"] is not None else 0
        tfidf = r["tfidf"] if r["tfidf"] is not None else 0.0
        pmi = r["pmi"] if r["pmi"] is not None else 0.0
        npmi = r["npmi"] if r["npmi"] is not None else 0.0
        ratio = r["ratio"] if r["ratio"] is not None else 0.0

        s_occ = r["s_occ"] if r["s_occ"] is not None else 1
        d_occ = r["d_occ"] if r["d_occ"] is not None else 1

        # Compute LLR
        k11 = coocc
        k12 = s_occ - coocc
        k21 = d_occ - coocc
        k22 = N_ds - d_occ - s_occ + coocc

        llr_val = compute_llr(k11, k12, k21, k22)

        # Accumulate
        coocc_list.append(coocc)
        tfidf_list.append(tfidf)
        pmi_list.append(pmi)
        npmi_list.append(npmi)
        llr_list.append(llr_val)
        ratio_list.append(ratio)

    #Step 4: Compute trimmed min and max for each metric
    coocc_min, coocc_max = trimmed_minmax(coocc_list)
    tfidf_min, tfidf_max = trimmed_minmax(tfidf_list)
    pmi_min, pmi_max = trimmed_minmax(pmi_list)
    npmi_min, npmi_max = trimmed_minmax(npmi_list)
    llr_min, llr_max = trimmed_minmax(llr_list)
    ratio_min, ratio_max = trimmed_minmax(ratio_list)

    coocc_min_full, coocc_max_full = min_max(coocc_list)
    tfidf_min_full, tfidf_max_full = min_max(tfidf_list)
    pmi_min_full, pmi_max_full = min_max(pmi_list)
    npmi_min_full, npmi_max_full = min_max(npmi_list)
    llr_min_full, llr_max_full = min_max(llr_list)
    ratio_min_full, ratio_max_full = min_max(ratio_list)

    # Step 5: Create bounds dictionary
    bounds_dict = {
        "coocc_min": coocc_min, "coocc_max": coocc_max,
        "tfidf_min": tfidf_min, "tfidf_max": tfidf_max,
        "pmi_min": pmi_min, "pmi_max": pmi_max,
        "npmi_min": npmi_min, "npmi_max": npmi_max,
        "llr_min": llr_min, "llr_max": llr_max,
        "ratio_min": ratio_min, "ratio_max": ratio_max
    }

    bounds_dict_full = {
        "coocc_min": coocc_min_full, "coocc_max": coocc_max_full,
        "tfidf_min": tfidf_min_full, "tfidf_max": tfidf_max_full,
        "pmi_min": pmi_min_full, "pmi_max": pmi_max_full,
        "npmi_min": npmi_min_full, "npmi_max": npmi_max_full,
        "llr_min": llr_min_full, "llr_max": llr_max_full,
        "ratio_min": ratio_min_full, "ratio_max": ratio_max_full
    }

    # Step 6: Store bounds in data/metric_bounds.json
    data_dir = ROOT_DIR / "data" / "preprocessed"
    data_dir.mkdir(exist_ok=True, parents=True)
    out_file = data_dir / "metric_bounds.json"
    out_file_full = data_dir / "metric_bounds_full.json"

    with open(out_file, "w") as f:
        json.dump(bounds_dict, f, indent=2)

    with open(out_file_full, "w") as f:
        json.dump(bounds_dict_full, f, indent=2)

    print(f"Metric bounds (trimmed 0.99) computed and saved to {out_file}")
    print(f"Full metric bounds (not trimmed) computed and saved to {out_file_full}")

if __name__ == "__main__":
    main()