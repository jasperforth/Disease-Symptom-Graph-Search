# backend/tests/test_search.py
"""
Contains tests for the search_multiple_unified endpoint in backend/app/api/search_multiple_unified.py.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_search_multiple_unified():
    payload = {
        "symptoms": [
            {"name": "Fever", "severity": "low"},
            {"name": "Cough", "severity": "medium"}
        ]
    }
    response = client.post("/searchMultiple", json=payload)
    assert response.status_code == 200, "Status code should be 200."
    
    data = response.json()
    assert "diseases" in data, "Response should contain 'diseases'."
    assert "related_symptoms" in data, "Response should contain 'related_symptoms'."
    assert isinstance(data["diseases"], list), "'diseases' should be a list."
    assert isinstance(data["related_symptoms"], list), "'related_symptoms' should be a list."
    assert len(data["diseases"]) > 0, "'diseases' list should not be empty."
    assert len(data["related_symptoms"]) > 0, "'related_symptoms' list should not be empty."

    for disease in data["diseases"]:
        assert "disease" in disease, "Each disease should have a 'disease' field."
        assert "score" in disease, "Each disease should have a 'score' field."
        assert isinstance(disease["disease"], str), "'disease' should be a string."
        assert isinstance(disease["score"], float), "'score' should be a float."
    
    for symptom in data["related_symptoms"]:
        assert isinstance(symptom, str), "Each related symptom should be a string."

    # Debug prints 
    print("Keys in response:", data.keys())
    print("Example disease:", data["diseases"][0])
    print("Examples related_symptoms:", data["related_symptoms"])
    print("Search Multiple Unified Test Passed")

if __name__ == "__main__":
    test_search_multiple_unified()
