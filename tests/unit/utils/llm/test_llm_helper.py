import os
import sys
import logging

from app.utils.llm import LLMHelper
from app.config import Config

logger = logging.getLogger(__name__)


# TODO: We should mock the OpenAI API in the unit tests

def test_get_llm():
    """Test get_llm."""

    config = Config()
    llm_helper = LLMHelper(config)
    llm = llm_helper.get_llm()

    assert llm is not None

def test_get_embeddings():
    """Test get_embeddings."""

    config = Config()
    llm_helper = LLMHelper(config)
    embeddings = llm_helper.get_embeddings()

    assert embeddings is not None
