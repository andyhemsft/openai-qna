import logging
import pytest

from app.config import Config
from app.utils.translator import AzureAITranslator


logger = logging.getLogger(__name__)


@pytest.fixture()
def azure_ai_translator():
    config = Config()
    return AzureAITranslator(config)


def test_translate(azure_ai_translator):
    text = "你好"
    translated_text = azure_ai_translator.translate(text)
    assert translated_text == "Hello"