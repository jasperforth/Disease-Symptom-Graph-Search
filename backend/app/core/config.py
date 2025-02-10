"""
./backend/app/core/config.py

This module contains the Settings class for managing application settings.
"""

from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

import dotenv
from fastapi.background import P

# TODO: Gather the settings and parameters from all over the codebase and put them here
# TODO: Clean up sensitive Data via .env

# determine the root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[3]

# load the .env file
dotenv_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path)

@dataclass
class Settings:
    # Neo4j settings
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    NEO4J_IMPORT_DIR: str = os.getenv("NEO4J_IMPORT_DIR")
    # Scoring settings
    # ALPHA: float = 0.35
    # BETA: float = 0.65
    # Logging settings
    LOG_DIR: str = os.getenv("LOG_DIR", ROOT_DIR / "logs")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Severity scaling
    SEVERITY_SCALING_LOW: float = 0.5
    SEVERITY_SCALING_MEDIUM: float = 1.0
    SEVERITY_SCALING_HIGH: float = 2.0

    # Disease-Service Scoring
    # Weights (global, local, base) should add up to 1, will be normalized anyway
    BASE_WEIGHT: float = 0.4
    LOCAL_WEIGHT: float = 0.2
    GLOBAL_WEIGHT: float = 0.4
    PARAM_COOCC: float = 0.4 # fraction for coocc vs tfidf=(1-cooc) in base

    # Toggles for global metrics (0 => off, 1 => on)
    # Will be normalized uniform to 1 if more than one is on
    PARAM_PMI: float = 0.0  # 0 or 1 to toggle PMI
    PARAM_NPMI: float = 1.0 # 0 or 1 to toggle NPMI
    PARAM_LLR: float = 0.0 # 0 or 1 to toggle LLR

    # Combination mode
    # "linear", "multiplicative_simple", "nonlinear_exponential", "nonlinear_sigmoid"
    MODE: str = "nonlinear_exponential"

    # TODO
    # Inlcude the params for
    # number of diseases
    # number of related symptoms shown (./api/search_multiple_unified.py) 
    # clamp value for ration and min max in disease service

settings = Settings()
# print(settings)
