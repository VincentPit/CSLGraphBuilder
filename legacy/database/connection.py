"""Database connection manager."""

from typing import Optional
from langchain_community.graphs import Neo4jGraph
from config import config
from exceptions import DatabaseError
from logger_config import logger


class DatabaseManager:
    """Manages Neo4j database connections."""
    
    def __init__(self):
        self._graph: Optional[Neo4jGraph] = None
    
    def get_graph(self) -> Neo4jGraph:
        """Get Neo4j graph connection."""
        if self._graph is None:
            self._graph = self._create_connection()
        return self._graph
    
    def _create_connection(self) -> Neo4jGraph:
        """Create a new Neo4j connection."""
        try:
            logger.info("Creating Neo4j database connection")
            graph = Neo4jGraph(
                url=config.database.neo4j_uri,
                username=config.database.neo4j_user,
                password=config.database.neo4j_password,
                database=config.database.neo4j_database
            )
            
            # Test connection
            result = graph.query("RETURN 1 as test")
            if not result:
                raise DatabaseError("Failed to connect to Neo4j database")
            
            logger.info("Successfully connected to Neo4j database")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise DatabaseError(f"Database connection failed: {e}") from e
    
    def close(self):
        """Close the database connection."""
        if self._graph is not None:
            try:
                self._graph._driver.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
            finally:
                self._graph = None
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            graph = self.get_graph()
            result = graph.query("RETURN 1 as test")
            return bool(result)
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()