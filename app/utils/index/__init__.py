from app.utils.index.indexing import *
from app.config import Config

def get_indexer(config: Config):
    """This function Returns the indexer based on the config.
    
    Args:
        config: the config object
        vector_store: the vector store
    Returns:
        the indexer
    """
    
    if config.CHUNKING_STRATEGY == 'fixed':
        return FixedChunkIndexer(config)

    else:
        raise ValueError('Indexer type not supported')
    
