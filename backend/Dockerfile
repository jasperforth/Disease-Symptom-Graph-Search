# backend/Dockerfile
# IMPORTANT: Build this image from the repository root so that the "data" folder is in the build context.
# For example, from the repository root run:
#   docker build -f backend/Dockerfile -t backend .
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file and create conda environment
COPY backend/environment.yml .
RUN conda env create -f environment.yml

ENV PATH=/opt/conda/envs/disease-symptom-app-py312/bin:$PATH

# Copy the entire content of the backend folder into /app/backend to preserve the package structure.
COPY backend /app/backend

# Expose port for the FastAPI server
EXPOSE 8000

# Run the FastAPI application using uvicorn.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
