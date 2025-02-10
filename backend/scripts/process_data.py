import dis
import sys
import os
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase
import logging

from backend.app.core.config import settings, ROOT_DIR
from backend.app.database.connection import Neo4jConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info(f'Base directory: {ROOT_DIR}')

    # File paths
    diseases_file = ROOT_DIR / 'data' / '41467_2014_BFncomms5212_MOESM1043_ESM.txt'
    symptoms_file = ROOT_DIR / 'data' / '41467_2014_BFncomms5212_MOESM1044_ESM.txt'
    relationships_file = ROOT_DIR / 'data' / '41467_2014_BFncomms5212_MOESM1045_ESM.txt'

    # Read data
    try:
        diseases = pd.read_csv(diseases_file, sep='\t')
        symptoms = pd.read_csv(symptoms_file, sep='\t')
        relationships = pd.read_csv(relationships_file, sep='\t')
        logging.info("Columns in relationships dataset: %s", relationships.columns.tolist())
    except Exception as e:
        logging.error(f'Error reading datafiles: {e}')
        sys.exit(1)

    # Validate data
    if diseases.isnull().values.any() or symptoms.isnull().values.any() or relationships.isnull().values.any():
        logging.error("Missing values found in datasets. Please check the data files.")
        sys.exit(1)

    # Compute global sums
    total_disease_occ = diseases['PubMed occurrence'].sum()
    total_symptom_occ = symptoms['PubMed occurrence'].sum()
    total_coocc = relationships['PubMed occurrence'].sum()
    logging.info(f"Total disease occurrence: {total_disease_occ}")
    logging.info(f"Total symptom occurrence: {total_symptom_occ}")
    logging.info(f"Total co-occurrence: {total_coocc}")

    # Rename columns
    diseases = diseases.rename(columns={'MeSH Disease Term': 'name', 'PubMed occurrence': 'pubmed_occurrence'})
    symptoms = symptoms.rename(columns={'MeSH Symptom Term': 'name', 'PubMed occurrence': 'pubmed_occurrence'})
    relationships = relationships.rename(columns={
        'MeSH Disease Term': 'disease_name', 
        'MeSH Symptom Term': 'symptom_name', 
        'PubMed occurrence': 'cooccurrence', 
        'TFIDF score': 'tfidf_score'
    })

    # Clean data
    diseases['display_name'] = diseases['name'].astype(str).str.strip()
    diseases['name'] = diseases['name'].astype(str).str.strip().str.lower()
    
    symptoms['display_name'] = symptoms['name'].astype(str).str.strip()
    symptoms['name'] = symptoms['name'].astype(str).str.strip().str.lower()

    relationships['disease_name'] = relationships['disease_name'].astype(str).str.strip().str.lower()
    relationships['symptom_name'] = relationships['symptom_name'].astype(str).str.strip().str.lower()

    # Visual check data 
    logging.info("Diseases dataset:")
    logging.info(diseases.head())
    logging.info("Symptoms dataset:")
    logging.info(symptoms.head())
    logging.info("Relationships dataset:")
    logging.info(relationships.head())

    # Initialize connection to Neo4j
    try:
        with Neo4jConnection() as conn:
            driver = conn.driver
            logging.info(f'Connection to Neo4j at {settings.NEO4J_URI} established successfully')

            # Determine Neo4j import directory
            neo4j_import_dir = Path(settings.NEO4J_IMPORT_DIR)
            logging.info(f'Neo4j import directory: {settings.NEO4J_IMPORT_DIR}')

            # Ensure the import directory exists
            if not neo4j_import_dir.exists():
                logging.error(f"Import directory {neo4j_import_dir} does not exist. Please check NEO4J_IMPORT_DIR environment variable.")  
                sys.exit(1)

            # Save to CSV files in Neo4j import directory
            diseases_csv = neo4j_import_dir / 'diseases.csv'
            symptoms_csv = neo4j_import_dir / 'symptoms.csv'
            relationships_csv = neo4j_import_dir / 'relationships.csv'

            logging.info(f"Saving diseases to {diseases_csv}")
            diseases[['name', 'display_name', 'pubmed_occurrence']].to_csv(diseases_csv, index=False, header=['name', 'display_name', 'pubmed_occurrence'])
            logging.info(f"Saving symptoms to {symptoms_csv}")
            symptoms[['name', 'display_name', 'pubmed_occurrence']].to_csv(symptoms_csv, index=False, header=['name', 'display_name', 'pubmed_occurrence'])
            logging.info(f"Saving relationships to {relationships_csv}")
            relationships[['disease_name', 'symptom_name', 'cooccurrence', 'tfidf_score']].to_csv(relationships_csv, index=False, header=['disease_name', 'symptom_name', 'cooccurrence', 'tfidf_score'])

            logging.info("Verifying CSV content for diseases.csv:")
            test_diseases_df = pd.read_csv(diseases_csv)
            logging.info(test_diseases_df.head().to_string())

            logging.info("Verifying CSV content for symptoms.csv:")
            test_symptoms_df = pd.read_csv(symptoms_csv)
            logging.info(test_symptoms_df.head().to_string())

            logging.info("Verifying CSV content for relationships.csv:")
            test_relationships_df = pd.read_csv(relationships_csv)
            logging.info(test_relationships_df.head().to_string())


            with driver.session() as session:
                # Create indexes for faster MERGE operations
                logging.info("Creating indexes...")
                session.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.name)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.name)")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Meta) REQUIRE m.name IS UNIQUE")
                logging.info("Indexes created successfully.")

                # Import Diseases
                logging.info("Importing diseases...")
                session.run(
                    """
                    CALL apoc.periodic.iterate(
                        'LOAD CSV WITH HEADERS FROM "file:///diseases.csv" AS row RETURN row',
                        'MERGE (d:Disease {name: toLower(row.name)})
                            SET d.display_name = row.display_name,
                                d.pubmed_occurrence = toInteger(row.pubmed_occurrence)',
                        {batchSize: 500, parallel: false}
                    )
                    """
                )
                logging.info("Diseases imported successfully.")

                # Import Symptoms
                logging.info("Importing symptoms...")
                session.run(
                    """
                    CALL apoc.periodic.iterate(
                        'LOAD CSV WITH HEADERS FROM "file:///symptoms.csv" AS row RETURN row',
                        'MERGE (s:Symptom {name: toLower(row.name)})
                            SET s.display_name = row.display_name,
                                s.pubmed_occurrence = toInteger(row.pubmed_occurrence)',
                        {batchSize: 500, parallel: false}
                    )
                    """
                )
                logging.info("Symptoms imported successfully.")

                # Import Relationships
                logging.info("Importing relationships...")
                session.run(
                    """
                    CALL apoc.periodic.iterate(
                        'LOAD CSV WITH HEADERS FROM "file:///relationships.csv" AS row RETURN row',
                        'MATCH (d:Disease {name: toLower(row.disease_name)}),
                            (s:Symptom {name: toLower(row.symptom_name)})
                        MERGE (d)-[r:HAS_SYMPTOM]->(s)
                            SET r.cooccurrence = toInteger(row.cooccurrence),
                                r.tfidf_score = toFloat(row.tfidf_score)',
                        {batchSize: 500, parallel: false}
                    )
                    """
                )
                logging.info("Relationships imported successfully.")

                # Compute degrees for Disease and Symptom nodes
                logging.info("Computing degrees for Disease and Symptom nodes...")
                # Updated query for Disease nodes
                session.run(
                    """
                    MATCH (d:Disease)
                    OPTIONAL MATCH (d)-[r]-()
                    WITH d, COUNT(r) AS degree
                    SET d.degree = coalesce(degree, 0)
                    """
                )
                # Updated query for Symptom nodes
                session.run(
                    """
                    MATCH (s:Symptom)
                    OPTIONAL MATCH (s)-[r]-()
                    WITH s, COUNT(r) AS degree
                    SET s.degree = coalesce(degree, 0)
                    """
                )
                logging.info("Degrees computed successfully.")

                # Compute probabilities P(d) and P(s)
                logging.info("Computing probabilities P(d) and P(s)...")
                session.run(
                    """
                    MATCH (d:Disease)
                    SET d.P_d = toFloat(d.pubmed_occurrence) / $total_disease_occ
                    """,
                    {'total_disease_occ': total_disease_occ}
                )
                session.run(
                    """
                    MATCH (s:Symptom)
                    SET s.P_s = toFloat(s.pubmed_occurrence) / $total_symptom_occ
                    """,
                    {'total_symptom_occ': total_symptom_occ}
                )
                logging.info("Probabilities P(d) and P(s) computed successfully.")

                # Compute joint probabilities P(d,s)
                logging.info("Computing joint probabilities P(d,s)...")
                session.run(
                    """
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]-(s:Symptom)
                    SET r.P_d_s = toFloat(r.cooccurrence) / $total_coocc
                    """,
                    {'total_coocc': total_coocc}
                )
                logging.info("Joint probabilities P(d,s) computed successfully.")

                # Compute local expectations and ratio
                logging.info("Computing local expectations and ratios...")
                session.run(
                    """
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]-(s:Symptom)
                    WHERE d.degree > 0 AND s.degree > 0
                    WITH d, s, r,
                        toFloat(d.pubmed_occurrence) AS occ_d,
                        toFloat(s.pubmed_occurrence) AS occ_s,
                        toFloat(d.degree) AS deg_d,
                        toFloat(s.degree) AS deg_s,
                        toFloat(r.cooccurrence) AS coocc,
                        toFloat(r.tfidf_score) AS tfidf_score
                    WITH d, s, r, occ_d, occ_s, deg_d, deg_s, coocc, tfidf_score,
                        (occ_d / deg_d) AS E_d,
                        (occ_s / deg_s) AS E_s
                    WITH d, s, r, coocc, tfidf_score, E_d, E_s,
                        sqrt(E_d * E_s) AS E
                    WITH d, s, r, coocc, tfidf_score, E,
                        (coocc / E) AS ratio
                    SET r.E = E,
                        r.ratio = ratio,
                        r.final_weight_local = tfidf_score * ratio
                    """
                )
                logging.info("Local expectations and ratios computed successfully.")

                # Compute PMI and NPMI
                logging.info("Computing PMI and NPMI...")
                session.run(
                    """
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]-(s:Symptom)
                    WHERE r.P_d_s IS NOT NULL AND d.P_d IS NOT NULL AND s.P_s IS NOT NULL
                    AND r.P_d_s > 0 AND d.P_d > 0 AND s.P_s > 0
                    WITH r, r.P_d_s AS P_d_s, d.P_d AS P_d, s.P_s AS P_s
                    WITH r, P_d_s, P_d, P_s,
                        log(P_d_s / (P_d * P_s)) AS PMI,
                        -log(P_d_s) AS H_d_s
                    WITH r, PMI, H_d_s,
                        CASE WHEN H_d_s <> 0 THEN PMI / H_d_s ELSE 0 END AS NPMI
                    SET r.PMI = PMI,
                        r.NPMI = NPMI
                    """
                )
                logging.info("PMI and NPMI computed successfully.")

                # Create Meta node with global sums
                logging.info("Creating Meta node with global sums...")
                session.run(
                    """
                    MERGE (m:Meta {name: 'GlobalStats'})
                    SET m.total_disease_occurrence = $tdo,
                        m.total_symptom_occurrence = $tso,
                        m.total_cooccurrence = $tcoocc
                    """,
                    {'tdo': int(total_disease_occ), 'tso': int(total_symptom_occ), 'tcoocc': int(total_coocc)}
                )
                logging.info("Meta node created successfully.")

            logging.info("Data imported and graph prepared successfully.")

    except Exception as e:
        logging.error(f'An error occurred during Neo4j operations: {e}')
        sys.exit(1)

logging.info("Data imported and Neo4j graph was prepared successfully.")


if __name__ == '__main__':
    main()
