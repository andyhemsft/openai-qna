import logging
import pytest

from app.config import Config
from app.utils.vectorstore.azuresearch import AzureSearch
from app.utils.llm import LLMHelper

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

@pytest.fixture
def azuresearch() -> AzureSearch:
    """Fixture for the AzureSearch class."""
    config = Config()

    embeddings = LLMHelper(config).get_embeddings()
    azuresearch = AzureSearch(config, embeddings)
    yield azuresearch

    if azuresearch.check_existing_index("test_index"):
        azuresearch.delete_index("test_index")

def test_similarity_search(azuresearch):
    """This function tests similarity search function for vector store."""

    # Special test case for Azure Search

    query = "匯豐卓越理財提供什麼服務"
    result = azuresearch.similarity_search(query, index_name='embeddings', search_text='HSBC Premier provide what services')

    assert len(result) > 0
    logger.info(result[0][0].page_content)