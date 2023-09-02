from app.utils.index.indexing import *
from app.config import Config
from app.utils.vectorstore import BaseVectorStore

def get_indexer(config: Config, vector_store: BaseVectorStore):
    """This function Returns the indexer based on the config.
    
    Args:
        config: the config object
        vector_store: the vector store
    Returns:
        the indexer
    """
    
    if config.INDEXER_TYPE == 'fixed_chunk':
        return FixedChunkIndexer(config, vector_store)

    else:
        raise ValueError('Indexer type not supported')
    
