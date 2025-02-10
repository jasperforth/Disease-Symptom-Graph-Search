.PHONY: init start-neo4j-database-dev recreate-dump build-image start-image stop-neo4j-container \
        restart-neo4j-container stop-docker neo4j-up start-backend start-frontend start-streamlit \
        start-full-app eval-experiments clean-all delete-all-images test-backend it-all-stop-and-go-away

##############################
# SECTION 1: FIRST UP        #
# Initialization & Population#
##############################
# Initialize environment: set permissions and create necessary directories.
init:
	./init_neo4j.sh init_environment

# Start Neo4j Database in Development Mode:
# - Initializes the environment and starts Neo4j.
# - Checks for a dump; if it exists, loads it; otherwise runs data processing.
# - At the end, checks for the metric bounds file.
#   If it does not exist, computes the metric bounds and prints a note.
#   - Inludes creating the backend container and running the metric bounds script.
#   	- Note: it takes a bit to create the conda environment and install the dependencies.
#   (NOTE: In the future, these metric bounds should be integrated into the Neo4j graph.)
start-neo4j-database-dev: init
	./init_neo4j.sh start_neo4j_database_dev
	@if [ ! -f "data/preprocessed/metric_bounds.json" ]; then \
	    echo "Metric bounds file not found. Running compute_metric_bounds.py..."; \
	    docker-compose run --rm backend python -m backend.scripts.compute_metric_bounds; \
	    echo "NOTE: The computed metric bounds should be integrated into the Neo4j graph in the future."; \
	else \
	    echo "Metric bounds file exists. Skipping computation."; \
	fi

# Recreate the database dump (if needed)
recreate-dump: init
	./init_neo4j.sh create_dump

##############################
# SECTION 2: ALL UP          #
# Starting Services After Initial Population
##############################
# Start the o Neo4j container.
neo4j-up:
	docker-compose up -d neo4j
	@echo "Neo4j container is up."

# Start the backend container (FastAPI app) and ensure metric bounds are computed.
start-backend:
	docker-compose up -d backend
	@if [ ! -f "data/preprocessed/metric_bounds.json" ]; then \
	    echo "Metric bounds file not found. Running compute_metric_bounds.py..."; \
	    docker-compose run --rm backend python -m backend.scripts.compute_metric_bounds; \
	    echo "NOTE: The computed metric bounds should be integrated into the Neo4j graph in the future."; \
	else \
	    echo "Metric bounds file exists. Skipping computation."; \
	fi

# Start the frontend container (React/Vite app).
start-frontend:
	docker-compose up -d frontend
	@echo "Frontend container is up."

# Composite target to start all main services (Neo4j, backend, and frontend).
# (Assumes that Neo4j is already up or populated.)
start-full-app: neo4j-up start-backend start-frontend
	@echo "Neo4j, backend, and frontend containers are up."

##############################
# SECTION 3: EXPERIMENTS/EVAL #
##############################
# Composite target to run experimental parameter grid search or launch evaluation.
# - First ensures that Neo4j is up.
# - If the experiment results file (default top_k=10) is not present,
#   runs the grid search experiment (which, for example, takes 20-30 minutes on an 8-core machine).
# - Otherwise, launches the Streamlit evaluation app.
# NOTE: By default, the parameter search uses all available CPU cores (n_jobs = -1 in parameter_search.py).
#       To limit the number of cores used, modify the n_jobs variable in the script.

# Start the Streamlit evaluation container.
start-streamlit:
	@ echo "Launching the Streamlit evaluation app..."
	docker-compose up -d streamlit

# Start the Streamlit evaluation container and run the parameter search if no results are found.
eval-experiments: 
	@if [ ! -f "data/experiments_param_search/parameter_search_comparison_topk_10.csv" ]; then \
	    echo "No experimental results found. Running grid search experiments (default top_k=10)..."; \
	    docker-compose run --rm backend python -m backend.scripts.parameter_search --top_k 10; \
		echo "Experimental results have been generated. Launching the Streamlit evaluation app..."; \
		docker-compose up -d streamlit;
	else \
	    echo "Experimental results found. Launching the Streamlit evaluation app..."; \
	    docker-compose up -d streamlit; \
	fi

# Run backend tests using pytest in the backend container.
test-backend: neo4j-up
	docker-compose run --rm backend pytest backend/tests/
	@echo "Backend tests have been run."

##############################
# SECTION 4: ALL DOWN        #
# Stopping & Cleaning Up     #
##############################
# Stop all Docker containers, networks, and volumes defined in docker-compose.yml.
all-down:
	docker-compose down --volumes --remove-orphans
	@echo "All Docker containers, networks, and volumes have been stopped."

# Remove all Docker images (ignoring errors if none exist)
delete-all-images:
	@docker rmi $$(docker images -q) 2>/dev/null || true
	@echo "All Docker images have been removed."

# Remove dangling volumes, networks, and caches
prune-all:
	@docker system prune -a --volumes -f
	@docker builder prune -a -f
	@echo "All Docker containers, networks, volumes, images, and build caches have been removed."

# Composite target: Stop containers, remove images, and prune all caches.
it-all-stop-and-go-away: all-down delete-all-images prune-all
	@echo "All Docker containers, networks, volumes, images, and caches have been stopped and removed."