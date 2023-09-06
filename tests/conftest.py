import pytest
from app.config import Config

# Default config for testing
Config.FAISS_LOCAL_FILE_INDEX = 'tests/unit/utils/vectorstore/test_faiss_local_file_index.faiss'

@pytest.fixture()
def client():

    from app import app 
    return app.test_client()