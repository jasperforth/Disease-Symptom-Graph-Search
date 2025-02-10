# backend/app/api/test_endpoint.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/test", response_model=str)
def test_endpoint():
    return "OK"
