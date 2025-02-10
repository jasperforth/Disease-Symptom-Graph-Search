#!/bin/bash
set -e

# This script manages loading dumps, running data processing, and creating dumps.

# Directory and file paths
PROJECT_DIR="$(pwd)"
DUMP_DIR="$PROJECT_DIR/data/neo4j_dumps"
SCRIPT_DIR="$PROJECT_DIR/backend/scripts"
CUSTOM_IMAGE_NAME="neo4j-prod:with-data"



# Function: Initialize environment
init_environment() {
    echo "Initializing environment..."
    chmod +x ./init_neo4j.sh ./backend/scripts/entrypoint.sh
    mkdir -p "$DUMP_DIR" "$PROJECT_DIR/data/neo4j_logs" "$PROJECT_DIR/data/neo4j_data" "$PROJECT_DIR/data/neo4j_import" "$PROJECT_DIR/data/experiments_param_search"
    echo "Environment initialized."
}

# Function: start Neo4j
start_neo4j() {
    echo "Starting Neo4j container..."
    docker-compose -f docker-compose.yml up -d neo4j
    echo "Waiting for Neo4j to initialize..."
    sleep 5
    echo "Neo4j container started."
}

# Function: stop Neo4j container
stop_neo4j_container() {
    echo "Stopping Neo4j container..."
    docker stop disease-symptom-neo4j-dev2 || echo "Failed to stop Neo4j container. Ensure it's running."
    
    echo "Waiting for Neo4j container to stop..."
    timeout=60
    elapsed=0
    while docker ps | grep -q "disease-symptom-neo4j-dev2"; do
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Timeout reached. Neo4j container did not stop in $timeout seconds."
            exit 1
        fi
        echo "Neo4j container is still running, waiting..."
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "Neo4j container stopped successfully."
}

# Function: restart Neo4j container
restart_neo4j_container() {
    echo "Restarting Neo4j container..."
    docker restart disease-symptom-neo4j-dev2
    echo "Waiting for Neo4j to initialize..."
    sleep 5
    echo "Neo4j container restarted successfully."
}

# Function: Run Data Processing Script
run_data_processing() {
    echo "Running data processing script..."
    eval "$(conda shell.bash hook)"
    conda activate disease-symptom-app-py312

    (
      cd "$PROJECT_DIR"
      python -m backend.scripts.process_data
    )

    # # Export the project root to PYTHONPATH
    # export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    # python "$SCRIPT_DIR/process_data.py"

    echo "Data processing complete."
}

# Function: (Re-)Create Dump
create_dump() {
    start_neo4j
    echo "Creating database dump from scratch..."
    export NEO4J_IMPORT_DIR="$PROJECT_DIR/data/neo4j_import"
    run_data_processing
    stop_neo4j_container 
    docker start disease-symptom-neo4j-dev2
    echo "Creating dump..."

    host_dump_path="$PROJECT_DIR/data/neo4j_dumps/neo4j.dump"
    # Check if dump exists on host, rename if it does
    if [ -f "$host_dump_path" ]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        new_dump_name="neo4j_${timestamp}.archive.dump"
        mv "$host_dump_path" "$PROJECT_DIR/data/neo4j_dumps/$new_dump_name"
        echo "Existing dump renamed to $new_dump_name"
    fi

    docker exec -it disease-symptom-neo4j-dev2 bash -c "/var/lib/neo4j/bin/neo4j-admin database dump neo4j --to-path=/data/neo4j_dump"
    restart_neo4j_container 
    echo "Database dump created in $DUMP_DIR."
}

# Function: Load Dump
load_dump() {
    echo "Loading Neo4j dump from $DUMP_DIR..."
    stop_neo4j_container 
    docker start disease-symptom-neo4j-dev2
    docker exec -it disease-symptom-neo4j-dev2 bash -c "neo4j-admin database load neo4j --from-path=/data/neo4j_dump --overwrite-destination=true"
    restart_neo4j_container 
    echo "Dump loaded successfully."
}

# Function: Start Neo4j Database in Development Mode
start_neo4j_database_dev() {
    echo "Starting Neo4j database in development mode..."
    start_neo4j
    if [ -f "$DUMP_DIR/neo4j.dump" ]; then
        echo "Found existing neo4j.dump, loading it..."
        load_dump
    else
        echo "No dump found. Creating dump by running data processing..."
        create_dump
    fi
    echo "Neo4j database started in development mode successfully."
}

# Function: Build Neo4j Custom Image from Dump
build_image() {
    if [ -f "$DUMP_DIR/neo4j.dump" ]; then
        echo "Found existing dump. Building custom Neo4j image..."
        docker build -f Dockerfile.neo4j -t "$CUSTOM_IMAGE_NAME" .
        echo "Custom Neo4j image built successfully."
    else
        echo "No dump found. Creating dump from scratch..."
        create_dump
        echo "Building custom Neo4j image after creating dump..."
        docker build -f Dockerfile.neo4j -t "$CUSTOM_IMAGE_NAME" .
        echo "Custom Neo4j image built successfully."
    fi
}

# Function: Start Neo4j using custom image, always recreating the container
start_image() {
    echo "Starting Neo4j using custom image (always recreating the container)..."
    start_neo4j
    # Stop and remove any existing container
    if [ "$(docker ps -aq -f name=disease-symptom-neo4j-prod2)" ]; then
        echo "Removing existing Neo4j container..."
        docker stop disease-symptom-neo4j-prod2 || true
        docker rm disease-symptom-neo4j-prod2 || true
    fi

    # Check if the custom image exists
    if docker image inspect "$CUSTOM_IMAGE_NAME" > /dev/null 2>&1; then
        echo "Custom image exists. Using it to recreate the container..."
    else
        echo "Custom image does not exist. Building it first..."
        build_image
    fi

    # Start the container with docker-compose to ensure auth is set
    echo "Starting the container using docker-compose.prod.yml..."
    docker-compose -f docker-compose.prod.yml up -d || {
        echo "Failed to start the container. Exiting."
        exit 1
    }

    echo "Neo4j container started successfully with custom image and authentication settings."
}



# Main: Calls the appropriate function based on the command-line arguments
case "$1" in
    init_environment)
        init_environment
        ;;
    start_neo4j_database_dev)
        start_neo4j_database_dev
        ;;
    create_dump)
        create_dump
        ;;
    build_image)
        build_image
        ;;
    start_image)
        start_image
        ;;
    stop_docker)
        stop_docker
        ;;
    restart_neo4j_container)
        restart_neo4j_container
        ;;
    *)
        echo "Usage: $0 {init_environment|start_neo4j_database_dev|create_dump|build_image|start_image|stop_docker}"
        exit 1
        ;;
esac