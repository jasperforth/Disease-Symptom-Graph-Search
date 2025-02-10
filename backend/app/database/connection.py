"""
./backend/app/database/connection.py

This module contains the Neo4jConnection class for connecting to the Neo4j database.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self._uri = os.getenv("NEO4J_URI")
        self._user = os.getenv("NEO4J_USER")
        self._password = os.getenv("NEO4J_PASSWORD")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    @property
    def driver(self):
        return self._driver
    
    def __enter__(self):
        """Enable 'with Neo4jConnection() as conn:' usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the driver is closed when exiting the 'with' block."""
        self.close()

    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print("Connected to Neo4j")

    def close(self):
        if self.driver:
            self.driver.close()
            print("Connection to Neo4j closed")

    def execute_query(self, query, parameters=None):
        if not self.driver:
            raise Exception("Driver not initialized!")
        try: 
            with self._driver.session() as session:
                result = session.run(query, parameters)
                return result.data()
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None
        
# Singleton pattern to ensure one connection instance
neo4j_conn = Neo4jConnection()