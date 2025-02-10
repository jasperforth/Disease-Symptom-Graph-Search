"""
./backend/app/services/disease_service.py

Provides services to retrieve and score diseases based on input symptoms.
Implements normalization of metrics using min-max with outlier smoothing.

TODO: Store normalization bounds in the Meta node instead of a JSON file in the future.
"""

import json
import math
from pathlib import Path
from typing import List

from backend.app.database.connection import neo4j_conn
from backend.app.models.schemas import DiseaseResponse, MultipleSymptomRequest
from backend.app.services.scoring import calculate_score
from backend.app.core.config import settings, ROOT_DIR
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

#############################
# 0) Helper Functions
#############################

def load_metric_bounds() -> dict:
    """
    Loads the min and max values for each metric from data/metric_bounds.json.
    TODO: In the future, fetch these bounds from the Meta node in Neo4j.
    """
    data_dir = ROOT_DIR / "data" / "preprocessed"
    bounds_file = data_dir / "metric_bounds.json"
    
    if not bounds_file.exists():
        logger.error(f"Metric bounds file not found at {bounds_file}. Please run compute_metric_bounds.py first.")
        raise FileNotFoundError(f"Metric bounds file not found at {bounds_file}.")
    
    with open(bounds_file, "r") as f:
        bounds = json.load(f)
    
    logger.debug(f"Loaded metric bounds from {bounds_file}")
    return bounds

def minmax_norm(x: float, x_min: float, x_max: float) -> float:
    """
    Applies min-max normalization to a value.
    Ensures the normalized value is within [0,1].
    """
    if x <= x_min:
        return 0.0
    if x >= x_max:
        return 1.0
    return (x - x_min) / (x_max - x_min)

def clamp_value(x: float, lower: float, upper: float) -> float:
    """
    Clamps x to the interval [lower, upper], without any normalization.
    """
    return max(lower, min(x, upper))

def compute_llr(k11: float, k12: float, k21: float, k22: float) -> float:
    """
    Computes the Log-Likelihood Ratio (LLR) for a contingency table.
    Since LLR is not stored in Neo4j, we compute it on-the-fly.

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
        The computed LLR value. Returns 0.0 in case of any computation error.
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

#############################
# 1) DiseaseService Class
#############################

class DiseaseService:
    def __init__(self, db=neo4j_conn):
        """
        Initializes the DiseaseService with a database connection and loads normalization bounds.
        """
        self._db = db
        self.bounds = load_metric_bounds()
    
    def get_diseases_by_symptom(
        self,
        payload: MultipleSymptomRequest,
        base_weight: float = settings.BASE_WEIGHT,
        local_weight: float = settings.LOCAL_WEIGHT,
        global_weight: float = settings.GLOBAL_WEIGHT,
        param_coocc: float = settings.PARAM_COOCC,
        param_pmi: float = settings.PARAM_PMI,  
        param_npmi: float = settings.PARAM_NPMI,  
        param_llr: float = settings.PARAM_LLR,
        mode: str = settings.MODE
    ) -> List[DiseaseResponse]:
        """
        Retrieves and scores diseases based on input symptoms using normalized metrics.
        
        Parameters
        ----------
        payload : MultipleSymptomRequest
            Contains the list of symptoms to match.
        smoothing_param : float
            Smoothing parameter for local ratio if needed.
        base_weight : float
            Weight for the base portion (cooccurrence + TF-IDF).
        local_weight : float
            Weight for the local ratio portion.
        global_weight : float
            Weight for the global metrics (PMI, NPMI, LLR).
        param_coocc : float
            Fraction of cooccurrence in the base portion (0..1).
        param_pmi : float
            Toggle for PMI (0..1).
        param_npmi : float
            Toggle for NPMI (0..1).
        param_llr : float
            Toggle for LLR (0..1).
        mode : str
            Combination mode for scoring.

        Returns
        -------
        List[DiseaseResponse]
            A list of up to 10 diseases, sorted by descending final score.
        """
        # Validate sum of weights
        total_weight = base_weight + local_weight + global_weight
        if abs(total_weight - 1.0) > 1e-7:
            logger.warning(f"Base, local, global weights sum to {total_weight}, expected 1.0.")

        # Determine if global metrics are active based on parameters
        is_global_active = param_pmi > 0 or param_npmi > 0 or param_llr > 0
        
        # Adjust global_weight based on active parameters
        adjusted_global_weight = global_weight if is_global_active else 0.0
        
        # Calculate the sum of active weights
        sum_active_weights = base_weight + local_weight + adjusted_global_weight
        
        if sum_active_weights > 0:
            # Normalize the active weights to sum to 1.0
            base_weight = base_weight / sum_active_weights
            local_weight = local_weight / sum_active_weights
            global_weight = adjusted_global_weight / sum_active_weights
            logger.debug(
                f"Adjusted weights - base: {base_weight:.3f}, "
                f"local: {local_weight:.3f}, global: {global_weight:.3f}"
            )
        else:
            # If no weights are active, default to base_weight = 1.0
            base_weight, local_weight, global_weight = 1.0, 0.0, 0.0
            logger.debug(
                "No active weights found. Defaulting to base_weight=1.0, "
                "local_weight=0.0, global_weight=0.0."
            )
        
        # Severity-based symptom scaling
        severity_map = {
            "low": settings.SEVERITY_SCALING_LOW,
            "medium": settings.SEVERITY_SCALING_MEDIUM,
            "high": settings.SEVERITY_SCALING_HIGH
        }

        # Prepare scaling for each symptom
        symptom_scaling = {}
        for symptom in payload.symptoms:
            severity = symptom.severity.lower()
            scaling = severity_map.get(severity, 1.0)
            symptom_scaling[symptom.name.lower()] = scaling
            logger.debug(f"Symptom: {symptom.name}, severity: {symptom.severity}, scaling: {scaling}")
        
        # Gather symptom names (lowercased)
        symptoms_lower = [s.name.lower() for s in payload.symptoms]
        logger.debug(f"Symptoms: {symptoms_lower}")

        # Query Neo4j for relevant edges
        query = """
        MATCH (s:Symptom)-[r:HAS_SYMPTOM]-(d:Disease)
        WHERE toLower(s.name) IN $symptoms_lower
          AND NOT toLower(d.name) IN $symptoms_lower
        RETURN 
            d.display_name AS disease_display_name,
            s.name AS symptom_name,
            d.pubmed_occurrence AS d_occ,
            d.degree AS d_deg,
            s.pubmed_occurrence AS s_occ,
            s.degree AS s_deg,
            r.cooccurrence AS cooccurrence, 
            r.tfidf_score AS tfidf_score,
            r.PMI AS pmi,
            r.NPMI AS npmi,
            r.ratio AS ratio
        LIMIT 300
        """
        results = self._db.execute_query(query, {"symptoms_lower": symptoms_lower})
        logger.debug(f"Number of relationships retrieved: {len(results)}")

        # Retrieve total_cooccurrence from Meta
        meta_q = """
        MATCH (m:Meta {name: 'GlobalStats'})
        RETURN m.total_cooccurrence AS total_coocc
        """
        meta_res = self._db.execute_query(meta_q, {})
        if not meta_res:
            logger.error("No 'GlobalStats' meta node found in Neo4j.")
            return []
        N_ds = meta_res[0]["total_coocc"]
        logger.debug(f"Global total_cooccurrence = {N_ds}")

        disease_scores = {}
        
        for r in results:
            # disease_name = r["disease_display_name"].lower()
            disease_display_name = r["disease_display_name"]
            symptom_name = r["symptom_name"].lower()

            # Retrieve raw metrics
            coocc_raw = r["cooccurrence"] or 0
            tfidf_raw = r["tfidf_score"] or 0.0
            ratio_raw = r["ratio"] or 0.0
            pmi_raw = r["pmi"] or 0.0
            npmi_raw = r["npmi"] or 0.0

            # Compute LLR on-the-fly
            s_occ = r["s_occ"] if r["s_occ"] else 1
            d_occ = r["d_occ"] if r["d_occ"] else 1
            k11 = coocc_raw
            k12 = s_occ - coocc_raw
            k21 = d_occ - coocc_raw
            k22 = N_ds - d_occ - s_occ + coocc_raw
            llr_raw = compute_llr(k11, k12, k21, k22)

            # Apply symptom scaling
            scale = symptom_scaling.get(symptom_name, 1.0)
            coocc_scaled = coocc_raw * scale
            tfidf_scaled = tfidf_raw * scale
            pmi_scaled = pmi_raw * scale
            npmi_scaled = npmi_raw * scale
            llr_scaled = llr_raw * scale

            # Min-max normalization
            coocc_norm = minmax_norm(coocc_scaled, self.bounds["coocc_min"], self.bounds["coocc_max"])
            tfidf_norm = minmax_norm(tfidf_scaled, self.bounds["tfidf_min"], self.bounds["tfidf_max"])
            pmi_norm = minmax_norm(pmi_scaled, self.bounds["pmi_min"], self.bounds["pmi_max"])
            npmi_norm = minmax_norm(npmi_scaled, self.bounds["npmi_min"], self.bounds["npmi_max"])
            llr_norm = minmax_norm(llr_scaled, self.bounds["llr_min"], self.bounds["llr_max"])

            # Clamp ratio 0.01 to 0.99 to avoid extreme values
            ratio_norm = clamp_value(ratio_raw, self.bounds["ratio_min"], self.bounds["ratio_max"])

            logger.debug(
                # f"Disease '{disease_name}', Symptom '{symptom_name}', "
                f"Disease Display Name '{disease_display_name}': "
                f"coocc_norm={coocc_norm:.3f}, tfidf_norm={tfidf_norm:.3f}, ratio_norm={ratio_norm:.3f}, "
                f"pmi_norm={pmi_norm:.3f}, npmi_norm={npmi_norm:.3f}, llr_norm={llr_norm:.3f}"
            )

            # Calculate score for this edge
            edge_score = calculate_score(
                coocc_norm=coocc_norm,
                tfidf_norm=tfidf_norm,
                ratio_norm=ratio_norm,
                pmi_norm=pmi_norm,
                npmi_norm=npmi_norm,
                llr_norm=llr_norm,
                base_weight=base_weight,
                local_weight=local_weight,
                global_weight=global_weight,
                param_coocc=param_coocc,
                param_pmi=param_pmi,
                param_npmi=param_npmi,
                param_llr=param_llr,
                mode=mode
            )

            disease_scores[disease_display_name] = disease_scores.get(disease_display_name, 0.0) + edge_score

        # Build the response objects
        diseases_list = []
        for d_display_name, sc in disease_scores.items():
            diseases_list.append(DiseaseResponse(disease=d_display_name, score=sc))

        # Final normalization so top disease = 1.0
        if diseases_list:
            max_score = max(d.score for d in diseases_list)
            if max_score > 0:
                for d in diseases_list:
                    d.score /= max_score
                logger.debug(f"Final normalization: highest score set to 1.0 (was {max_score:.4f}).")

        # Sort by descending score, return top 10
        diseases_sorted = sorted(diseases_list, key=lambda x: x.score, reverse=True)
        return diseases_sorted[:10]