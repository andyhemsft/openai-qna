import logging
import pytest

from app.config import Config
from app.utils.conversation.intent import LLMIntentDetector


logger = logging.getLogger(__name__)


@pytest.fixture
def intent_detector():
    config = Config()
    return LLMIntentDetector(config)


def test_intent_detector(intent_detector):

    # Test case 1
    question = "Who is Elon Musk?"

    intent = intent_detector.detect_intent(question)

    assert intent == "DocRetrieval"


    # Test case 2
    question = "What is the sales of Agent B?"

    intent = intent_detector.detect_intent(question)

    assert intent == "MDRT_QnA"