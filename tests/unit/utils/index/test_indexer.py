import logging
import pytest

from app.config import Config
from app.utils.llm import LLMHelper
from app.utils.vectorstore import get_vector_store
from app.utils.index.indexing import FixedChunkIndexer
from app.utils.index import get_indexer

logger = logging.getLogger(__name__)


@pytest.fixture()
def vector_store():
    """This function returns a faiss vector stores."""

    # Load config
    config = Config()

    llm_helper = LLMHelper(config)
    embeddings = llm_helper.get_embeddings()

    vector_store = get_vector_store(config, embeddings)

    yield vector_store


def test_get_indexer(vector_store):
    """This function tests get indexer function."""

    # Load config
    config = Config()

    # Save old config
    old_chunking_strategy = Config.CHUNKING_STRATEGY

    # Load Fixed Chunk Indexer
    Config.CHUNKING_STRATEGY = 'fixed'
    indexer = get_indexer(config, vector_store)

    assert isinstance(indexer, FixedChunkIndexer)

    # Restore old config
    Config.CHUNKING_STRATEGY = old_chunking_strategy

@pytest.fixture()
def indexer(vector_store):
    """Get all indexers."""

    indexer = {}

    # Load config
    config = Config()

    # Save old config
    old_chunking_strategy = Config.CHUNKING_STRATEGY

    # Load Fixed Chunk Indexer
    Config.CHUNKING_STRATEGY = 'fixed'
    indexer['fixed'] = get_indexer(config, vector_store)

    # Restore old config
    Config.CHUNKING_STRATEGY = old_chunking_strategy

    yield indexer

def test_create_index(indexer):
    """This function tests create index function."""

    pass

def test_drop_index(indexer):
    """This function tests drop index function."""

    pass

def test_add_document(indexer):
    """This function tests add document function."""

    pass

def test_similarity_search(indexer):
    """This function tests similarity search function."""

    pass