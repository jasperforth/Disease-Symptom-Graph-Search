"""
./backend/app/models/schemas.py

This file contains the Pydantic models for the API request and response bodies.
"""

from pydantic import BaseModel
from typing import List

class DiseaseResponse(BaseModel):
    disease: str
    score: float

class SymptomSeverity(BaseModel):
    name: str
    severity: str # Example: low, medium, high

class MultipleSymptomRequest(BaseModel):
    symptoms: List[SymptomSeverity]

class CombinedResponse(BaseModel):
    diseases: List[DiseaseResponse]
    related_symptoms: List[str]