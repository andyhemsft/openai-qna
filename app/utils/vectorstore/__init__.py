from typing import List

from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from app.utils.vectorstore.base import BaseVectorStore
from app.utils.vectorstore.faiss import FAISSExtended
from app.utils.vectorstore.redis import RedisExtended
from app.config import Config

def get_vector_store(config: Config, embeddings: Embeddings) -> BaseVectorStore:
    """This function Returns the vector store based on the config.
    
    Args:
        config: the config object
        embeddings: the embeddings model
    Returns:
        the vector store
    """

    if config.VECTOR_STORE_TYPE == 'faiss':
        vector_store = FAISSExtended(config, embeddings)

    elif config.VECTOR_STORE_TYPE == 'redis':
        vector_store = RedisExtended(config, embeddings)

    else:
        raise ValueError('Vector store type not supported')

    return vector_store
