"""
./backend/app/main.py

This module initializes the FastAPI application and includes the API routers.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import (
    symptom_suggestions, 
    search_multiple_unified,
    test_endpoint 
    )
from backend.app.utils.logger import get_logger
from backend.app.database.connection import neo4j_conn
from contextlib import asynccontextmanager

logger = get_logger(__name__)

# start Neo4j database
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Connecting to Neo4j")
    try:
        yield
    finally:
        logger.info("Application shutdown: Closing Neo4j connection")
        neo4j_conn.close()


app = FastAPI(
    title="Disease-Symptom Analysis API",
    description="API for analyzing medical symptoms and their connections to diseases.",
    version="1.0.0",
    lifespan=lifespan,
)

# Cinfigure CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # allow these origins
    allow_credentials=True,
    allow_methods=["*"], # allow all HTTP methods
    allow_headers=["*"], # allow all headers
)

# include API routers
# app.include_router(search.router, tags=["Search"])
app.include_router(search_multiple_unified.router, tags=["Search Multiple Unified"]) 
# app.include_router(search_multiple.router, tags=["Search Multiple"]) # Replaced by Search Multiple Unified
app.include_router(symptom_suggestions.router, tags=["Symptom Suggestions"])
app.include_router(test_endpoint.router, tags=["Test"])

