import os
import logging
import pytest

from langchain.docstore.document import Document

from app.utils.llm import LLMHelper
from app.utils.vectorstore import get_vector_store
from app.utils.vectorstore.faiss import FAISSExtended
from app.utils.vectorstore.redis import RedisExtended
from app.config import Config

logger = logging.getLogger(__name__)


def test_get_vector_store():
    """This function tests get vector store function."""

    # Load config
    config = Config()

    # Save old config
    old_vector_store_type = Config.VECTOR_STORE_TYPE

    llm_helper = LLMHelper(config)
    embeddings = llm_helper.get_embeddings()

    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store = get_vector_store(config, embeddings)

    assert isinstance(vector_store, FAISSExtended)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    vector_store = get_vector_store(config, embeddings)

    assert isinstance(vector_store, RedisExtended)


    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type

@pytest.fixture()
def vector_store():
    """This function returns all supported vector stores."""

    vector_store = {}

    # Load config
    config = Config()

    # Save old config
    old_vector_store_type = Config.VECTOR_STORE_TYPE

    llm_helper = LLMHelper(config)
    embeddings = llm_helper.get_embeddings()

    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store['faiss'] = get_vector_store(config, embeddings)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    vector_store['redis'] = get_vector_store(config, embeddings)

    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type

    yield vector_store

def test_add_documents(vector_store):
    """This function tests add documents function for vector store."""

    doc =  Document(page_content="This is a test document only.", metadata={"source": "local"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc])

def test_similarity_search(vector_store):
    """This function tests similarity search function for vector store."""

    query = "This is a test query only."
    doc =  Document(page_content="This is a test document only.", metadata={"source": "local"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc])
            result = vector_store.similarity_search(query)

            assert len(result) > 0
            assert result[0][0].page_content == "This is a test document only."


    