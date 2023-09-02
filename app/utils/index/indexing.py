import os

from typing import Any
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter

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

    def create_index(self):

    def add_document(self, source_url: str, **kwargs: Any):
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
        Returns:
            the embedded text
        """

        return None

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

    def add_documents(self, source_url: str, **kwargs: Any) -> None:
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
        Returns:
            the embedded text
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
        self.vector_store.add_document(document)

        return None