
# Disease-Symptom Graph Application

> **Note:** This project is a development prototype.

## Overview

This project analyzes relationships between diseases and symptoms using PubMed data. It provides:

- **Graph Database (Neo4j):**  
  Stores disease and symptom nodes with relationships weighted by metrics (e.g., TF-IDF, co-occurrence, PMI, NPMI, LLR).

- **Backend (FastAPI):**  
  Processes queries, computes composite disease scores from normalized metrics, and exposes a REST API.

- **Frontend (React):**  
  Offers an interactive UI for entering symptoms, ranking diseases, and displaying related suggestions.

- **Experimental Framework:**  
  Includes grid search (for parameter tuning) and metric normalization. By default, the grid search uses all CPU cores (n_jobs = -1). Adjust in `backend/scripts/parameter_search.py` if desired.

---

## Repository Layout

```plaintext
DBMS_project5_ws2425/
├── backend/                 # FastAPI backend, scripts, tests
│   ├── app/                 # Application modules (routers, models, etc.)
│   ├── Dockerfile           # Build the backend image
│   ├── environment.yml      # Conda environment definition
│   ├── scripts/             # Data processing & experiments (parameter_search.py, etc.)
│   └── tests/               # Backend tests (pytest)
├── data/
│   ├── experiments_param_search/  # Experimental results (ignored in Git)
│   ├── neo4j_data/          # Neo4j persistent data (ignored)
│   ├── neo4j_logs/          # Neo4j logs (ignored)
│   └── logs/                # Backend logs (ignored)
├── docker-compose.yml       # Docker Compose config for official Neo4j, backend, frontend
├── docker-compose.prod.yml  # Optional Docker Compose for custom Neo4j image
├── init_neo4j.sh            # Shell script for advanced dev tasks (Make approach references this)
├── Makefile                 # Shortcut commands for Mac/Linux
├── .env.example             # Example environment variables
└── README.md                # This file
```

---

## Environment Setup

1. **Docker & Docker Compose** – Required for containerized Neo4j, backend, and frontend.  
2. **Node.js 20.18.1** & **npm 9.6.7** – For building and running the React frontend.  
3. **(Optional) Miniconda/Anaconda** – Only if you plan to run Python scripts locally instead of Docker.  
4. **Create a `.env` File** – Based on `.env.example`, for example:
   ```dotenv
   NEO4J_AUTH=neo4j/your_secret
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_secret
   NEO4J_IMPORT_DIR=/var/lib/neo4j/import
   LOG_DIR=./data/logs/backend
   ```

---

## How to Run

Below are two approaches for each step.

### 1. Create & Populate Neo4j

**Option A (Makefile)**  
```bash
make start-neo4j-database-dev
```
- Initializes environment (folders, permissions)  
- Starts Neo4j and checks if a dump exists; if not, runs data processing to create one  
- Checks if metric bounds exist; if not, computes them

**Option B (Manual)**  
1. Create necessary directories:
   ```bash
   mkdir -p data/neo4j_data data/neo4j_logs data/neo4j_import data/experiments_param_search
   chmod +x backend/scripts/entrypoint.sh
   ```
2. Start Neo4j:
   ```bash
   docker-compose up -d neo4j
   ```
3. Check if a database dump exists; if not, run:
   ```bash
   docker-compose run --rm backend python -m backend.scripts.process_data
   docker exec -it disease-symptom-neo4j-dev2 bash -c "neo4j-admin database dump neo4j --to-path=/data/neo4j_dump"
   ```
4. If metric bounds are missing:
   ```bash
   docker-compose run --rm backend python -m backend.scripts.compute_metric_bounds
   ```

### 2. Start All Services

**Option A (Makefile)**  
```bash
make start-full-app
```
- Starts Neo4j (neo4j-up), then the backend and frontend containers.

**Option B (Manual)**  
```bash
docker-compose up -d neo4j
docker-compose up -d backend
docker-compose up -d frontend
```

### 3. Experimental Grid Search & Streamlit

**Option A (Makefile)**  
```bash
make eval-experiments
```
- If no existing results, runs the grid search (top_k=10).  
- Otherwise, launches the Streamlit container.

**Option B (Manual)**  
1. **Run the parameter search**:
   ```bash
   docker-compose run --rm backend python -m backend.scripts.parameter_search --top_k 10
   ```
2. **Launch the Streamlit evaluation**:
   ```bash
   docker-compose up -d streamlit
   ```

### 4. Stopping & Cleaning

**Option A (Makefile)**

- Stop containers (and remove volumes/networks):
  ```bash
  make all-down
  ```
- Remove all images:
  ```bash
  make delete-all-images
  ```
- Or do both:
  ```bash
  make it-all-stop-and-go-away
  ```

**Option B (Manual)**  
```bash
docker-compose down --volumes --remove-orphans
docker images -q | xargs -r docker rmi
```
*(Use caution if you have other containers/images.)*

---

## Testing

**Option A (Makefile)**  
```bash
make test-backend
```
*(Requires Neo4j up.)*

**Option B (Manual)**  
```bash
docker-compose run --rm backend pytest backend/tests/
```

---

## Further Notes

- **Conda Environment (Optional):**  
  If you prefer to run scripts locally, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) and:
  ```bash
  conda env create -f backend/environment.yml
  conda activate disease-symptom-app-py312
  ```
- **CPU Usage in Grid Search:**  
  By default, the parameter search script uses all CPU cores (n_jobs=-1). Change `n_jobs` in `backend/scripts/parameter_search.py` if needed.  
- **Windows Path Differences:**  
  On Windows, replace `./` with `.\` and ensure Docker Desktop is properly configured for volume mounts.

---

## Contact

For questions or suggestions, please open an issue or contact the maintainer.
