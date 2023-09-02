from typing import List, Optional, Dict, Any, Tuple
from abc import abstractmethod

from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from app.config import Config

class BaseVectorStore:
    """This class represents a Base Vector Store."""

    def __init__(self, config: Config, embeddings: Embeddings):
        """
        Initialize the Base Vector Store.

        Args:
            config: the config object
            embeddings: the embeddings model
        """

        self.config = config
        self.embeddings = embeddings


    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """This function adds documents to the vector store.
        
        Args:
            documents: the documents to add
        Returns:
            none
        """

    @abstractmethod
    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """

    @abstractmethod
    def create_index(self, index_name: str) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """