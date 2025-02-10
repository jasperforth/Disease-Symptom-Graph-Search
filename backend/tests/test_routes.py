# backend/tests/test_routes.py
"""
Contains tests for the test endpoint from backend/app/api/test_routes.py.
"""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_test_endpoint():
    response = client.get("/test")
    assert response.status_code == 200, "Status code should be 200."
    assert response.json() == "OK", "The test endpoint should return 'OK'."
    print("Test endpoint returned:", response.json())

if __name__ == "__main__":
    test_test_endpoint()
