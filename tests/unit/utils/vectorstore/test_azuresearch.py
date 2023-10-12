import logging
import pytest

from app.config import Config
from app.utils.vectorstore.azuresearch import AzureSearch
from app.utils.llm import LLMHelper

from langchain.docstore.document import Document


@pytest.fixture
def azuresearch() -> AzureSearch:
    """Fixture for the AzureSearch class."""
    config = Config()

    embeddings = LLMHelper(config).get_embeddings()
    azuresearch = AzureSearch(config, embeddings)
    yield azuresearch

    if azuresearch.check_existing_index("test_index"):
        azuresearch.delete_index("test_index")

def test_check_existing_index(azuresearch: AzureSearch) -> None:
    """Test the check_existing_index function."""
    assert azuresearch.check_existing_index("test_index") == False

def test_create_and_delete_index(azuresearch: AzureSearch) -> None:
    """Test the create_index and drop_index function."""

    metadata_schema = {
        "source": "TEXT",
        "chunk": "NUMERIC",
    }

    azuresearch.create_index("test_index", metadata_schema=metadata_schema)
    assert azuresearch.check_existing_index("test_index") == True

    azuresearch.drop_index("test_index")
    assert azuresearch.check_existing_index("test_index") == False


def test_add_documents(azuresearch: AzureSearch) -> None:

    pass