import os
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.docstore.document import Document

from app.utils.vectorstore import BaseVectorStore
from app.config import Config

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
                document = TextLoader(source_url).load()
            else:
                document = WebBaseLoader(source_url).load()
        except:
            raise ValueError('Invalid source url')

        # Split the document into chunks
        chunks = self.text_splitter.split(document)

        # Add the chunks to the vector store
        self.vector_store.add_document(chunks, index_name=index_name, **kwargs)

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
    