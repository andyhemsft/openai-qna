import os
import logging
import pytest
import shutil

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


    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store = get_vector_store(config)

    assert isinstance(vector_store, FAISSExtended)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    vector_store = get_vector_store(config)

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

    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    vector_store['faiss'] = get_vector_store(config)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    vector_store['redis'] = get_vector_store(config)

    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type

    yield vector_store

    # Tear down
    if os.path.exists(config.FAISS_LOCAL_FILE_INDEX):
        shutil.rmtree(config.FAISS_LOCAL_FILE_INDEX)

def test_load_local(vector_store):
    """This function tests load local function for vector store."""

    config = Config()



    for key, vector_store in vector_store.items():
        if key == 'faiss':
            logger.info('Testing FAISS load from file')

            assert os.path.exists(config.FAISS_LOCAL_FILE_INDEX) == False
            vector_store.load_local(config.FAISS_LOCAL_FILE_INDEX)

            assert os.path.exists(config.FAISS_LOCAL_FILE_INDEX) == True
            vector_store.load_local(config.FAISS_LOCAL_FILE_INDEX)

            shutil.rmtree(config.FAISS_LOCAL_FILE_INDEX)


def test_add_documents(vector_store):
    """This function tests add documents function for vector store."""

    doc =  Document(page_content="This is a test document only.", metadata={"source": "local"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc])

def test_add_texts(vector_store):
    """This function tests add texts function for vector store."""

    text = "This is a test document only."
    metadata = {"source": "local"}

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_texts([text], [metadata])

def test_similarity_search(vector_store):
    """This function tests similarity search function for vector store."""

    query = "This is a test query only."
    doc1 =  Document(page_content="This is a test document from local.", metadata={"source": "local"})
    # doc2 =  Document(page_content="This is a test document from web.", metadata={"source": "web"})

    for key, vector_store in vector_store.items():
        if key == 'faiss':
            vector_store.add_documents([doc1])
            vector_store.add_texts(["This is a test document from web."], [{"source": "web"}])
            result = vector_store.similarity_search(query, filter={"source": "web"})

            assert len(result) > 0
            assert result[0][0].page_content == "This is a test document from web."
            assert result[0][0].metadata["source"] == "web"


    