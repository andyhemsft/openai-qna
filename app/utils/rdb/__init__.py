import pyodbc
import logging
from typing import List, Any

from app.config import Config

logger = logging.getLogger(__name__)

class ResultSet:

    def __init__(self, results: list) -> None:
        pass

class RelationDB:

    def __init__(self, config: Config) -> None:

        self.connection_str = config.SQL_CONNECTION_STRING
        self.conn = pyodbc.connect(self.connection_str)


    def execute(self, query: str) -> List[Any]:
        """Execute a query and return the results.
        
        Args:
            query (str): the query to execute

        Returns:
            List[Any]: the results of the query    
        """

        result = []

        cursor = self.conn.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        for r in records:
            result.append(r)

        return result