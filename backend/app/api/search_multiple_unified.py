"""
./backend/app/api/search_multiple_unified.py

This module contains the FastAPI router for the unified search endpoint 
for the disease search as well as the related symptoms suggestion.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from backend.app.models.schemas import MultipleSymptomRequest, CombinedResponse
from backend.app.services.disease_service import DiseaseService
from backend.app.database.connection import neo4j_conn
from backend.app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/searchMultiple", response_model=CombinedResponse) 
def search_multiple_unified(payload: MultipleSymptomRequest) -> CombinedResponse:
    """
    1. Get top diseases (up to 10) via DiseaseService.
    2. With those top diseases, find related symptoms, using a simple combined
       (coocc * 0.4 + tfidf * 0.6) ranking. Limit to 10.
    3. Return both disease list and related symptoms in one JSON.
    """
    # DISEASE SCORING
    logger.info(f"Received payload: {payload.model_dump()}")
    try:
        # 1) DISEASE SCORING
        service = DiseaseService()
        diseases = service.get_diseases_by_symptom(payload)
        logger.info(f"Diseases search results: {diseases}")
        if not diseases:
            logger.warning("No diseases found for the given symptoms.")
            return CombinedResponse(diseases=[], related_symptoms=[])
        
        # Prepare final diseases (the top k already limited by get_diseases_by_symptom)
        top_disease_names_lower = [d.disease.lower() for d in diseases]  # collect display_name
        symptoms_lower = [sym.name.lower() for sym in payload.symptoms]

        # Query for candidate symptoms, bridging via diseases, 
        # limit and sort by combined score 0.4*cooccurrence + 0.6*tfidf
        # sum over scores, to get the best combined for the full payload set.
        query = """
        MATCH (d:Disease)-[r:HAS_SYMPTOM]-(s:Symptom)
        WHERE d.name IN $top_disease_names_lower
        AND NOT s.name IN $symptoms_lower
        WITH s, sum(0.4*r.cooccurrence + 0.6*r.tfidf_score) AS combined
        ORDER BY combined DESC
        LIMIT 12
        RETURN s.display_name AS candidate_name, combined
        """

        results = neo4j_conn.execute_query(query, {
            "top_disease_names_lower": top_disease_names_lower,
            "symptoms_lower": symptoms_lower
        })
        logger.debug(f"Related symptoms query results: {results}")

        related_symptoms = [r["candidate_name"] for r in results]

        response = CombinedResponse(diseases=diseases, related_symptoms=related_symptoms)
        logger.info(f"Unified search results: {response}")
        
        return response

    except HTTPException as http_exc:
        raise http_exc  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in /searchMultiple: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
