import os
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
import logging

from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.docstore.document import Document

from app.utils.vectorstore import BaseVectorStore
from app.config import Config

logger = logging.getLogger(__name__)

class Indexer:
    """This class represents an Indexer."""

    def __init__(self, config: Config, vector_store: BaseVectorStore):
        """
        Initialize the Indexer.

        Args:
            config: the config object
            vector_store: the vector store
        """

        self.config = config
        self.vector_store = vector_store

    @abstractmethod
    def create_index(self, index_name: str) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

    @abstractmethod
    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

    @abstractmethod
    def add_document(self, source_url: str, index_name: str, **kwargs: Any) -> None:
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
            index_name: the index name
        Returns:
            none
        """

class FixedChunkIndexer(Indexer):
    """This class represents a Fixed Chunk Indexer."""

    def __init__(self, config: Config, vector_store: BaseVectorStore):
        """
        Initialize the Fixed Chunk Indexer.

        Args:
            config: the config object
            vector_store: the vector store
        """

        super().__init__(config, vector_store)
        self.chunk_size = config.CHUNKING_STRATEGY_MAX_LENGTH
        self.chunk_overlap = config.CHUNKING_STRATEGY_OVERLAP

        assert self.chunk_size > 0
        assert self.chunk_overlap >= 0

        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def add_document(self, source_url: str, index_name: str, **kwargs: Any) -> None:
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
            index_name: the index name
        Returns:
            none
        """

        try:
            # Check if source url is a file
            if os.path.isfile(source_url):
                logging.debug(f'Loading document from file {source_url}')
                document = TextLoader(source_url, encoding = 'utf-8').load()
            else:
                logging.debug(f'Loading document from web {source_url}')
                document = WebBaseLoader(source_url).load()
        except Exception as e:
            logger.error(e)
            raise e
            

        # Split the document into chunks
        chunks = self.text_splitter.split_documents(document)

        keys = []
        for i, chunk in enumerate(chunks):
            # Create a unique key for the chunk
            source_url = source_url.split('?')[0]

            chunk.metadata = {"source": source_url, "chunk_id": i}

        # Add the chunks to the vector store
        self.vector_store.add_documents(chunks, index_name=index_name, **kwargs)

        return None
    
    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        """
        return self.vector_store.similarity_search(query, k, filter=filter, index_name=index_name)
    
    def create_index(self, index_name: str) -> None:
        """This function creates an index.
        
        Args:  
            index_name: the index name
        Returns:
            none
        """

        return self.vector_store.create_index(index_name)
    
    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        return self.vector_store.drop_index(index_name)
    