"""
./backend/app/api/symptom_suggestions.py

This module contains the FastAPI router for symptom suggestions
based on fuzzy search over symptom names in the Neo4j database.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List

from backend.app.database.connection import neo4j_conn
from backend.app.utils.logger import get_logger

router = APIRouter()

logger = get_logger(__name__)

@router.get("/symptomsSuggestions", response_model=List[str])
def symptom_suggestions(query: str = Query(..., min_length=1, max_length=50)):
    """
    Returns a list of symptom names that start with the given query (case-insensitive).
    """
    try:
        query_lower = query.lower()
        cypher_query = """
        MATCH (s:Symptom)
        WHERE toLower(s.name) STARTS WITH $query_lower
        RETURN s.display_name AS name
        LIMIT 10
        """
        results = neo4j_conn.execute_query(cypher_query, {"query_lower": query_lower})
        suggestions = [r["name"] for r in results]
        logger.info(f"Suggestions for query '{query}': {suggestions}")
        return suggestions
    except Exception as e:
        logger.error(f"Error in /symptoms/suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")