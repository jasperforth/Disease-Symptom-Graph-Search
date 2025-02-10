"""
./backend/tests/test_service.py

Contains tests for the DiseaseService class in backend/app/services/disease_service.py.
"""
import pytest 
from backend.app.services.disease_service import DiseaseService
from backend.app.models.schemas import DiseaseResponse, MultipleSymptomRequest, SymptomSeverity

def test_get_disease_by_symptom():
    service = DiseaseService()  # db is set to default neo4j_conn
    payload = MultipleSymptomRequest(symptoms=[
        SymptomSeverity(name="Fever", severity="low"),
        SymptomSeverity(name="Cough", severity="medium")
    ])
    diseases = service.get_diseases_by_symptom(payload)
    assert isinstance(diseases, list), "Response should be a list."
    assert len(diseases) > 0, "Response should not be empty."   
    for disease in diseases:
        assert isinstance(disease, DiseaseResponse), "Each item should be a DiseaseResponse instance."
        assert isinstance(disease.disease, str), "Disease name should be a string."
        assert isinstance(disease.score, float), "Score should be a float."

    print("Get Disease by Symptom Test Passed")
    print(f"Number of diseases: {len(diseases)}")
    print(f"Example diseases: {diseases[0].disease}, {diseases[-1].disease}")
    print(f"Example scores: {diseases[0].score}, {diseases[-1].score}")


if __name__ == "__main__":
    test_get_disease_by_symptom()
