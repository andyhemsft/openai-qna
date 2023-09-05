
import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as FAISS_TYPE

from app.utils.vectorstore.base import BaseVectorStore
from app.config import Config

logger = logging.getLogger(__name__)

class FAISSExtended(BaseVectorStore):
    """This class represents a FAISS Vector Store."""

    def __init__(self, config: Config, embeddings: Embeddings):
        """
        Initialize the FAISS Vector Store.

        Args:
            config: the config object
            embeddings: the embeddings model
        """

        super().__init__(config, embeddings)

        texts = ["FAISS"]
        self.vector_store = FAISS_TYPE.from_texts(texts, embeddings)


    def add_documents(
            self, 
            documents: List[Document], 
            index_name: Optional[str] = None,
            **kwargs: Any
        ) -> None:
        """This function adds documents to the vector store.
        
        Args:
            documents: the documents to add
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support index_name parameter')
        self.vector_store.add_documents(documents)

    def add_texts(
            self, 
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None, 
            index_name: str = None, 
            **kwargs: Any
        ) -> None:
        """This function adds texts to the vector store.
        
        Args:
            texts: the texts to add
            index_name: the index name
        Returns:
            none
        """

        self.vector_store.add_texts(texts, metadatas=metadatas)

    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """

        return self.vector_store.similarity_search_with_relevance_scores(query, k, filter=filter)

    def create_index(self, index_name: str) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support creating indexes')

    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        logger.warning('FAISS does not support dropping indexes')

