import logging
import pytest

from app.config import Config
from app.utils.conversation.bot import LLMChatBot
from app.utils.conversation.history import HistoryManager, HistoryStore

logger = logging.getLogger(__name__)


@pytest.fixture()
def llm_chat_bot():
    config = Config()

    history_store = HistoryStore(config)
    history_manager = HistoryManager(config, history_store)
    llm_chat_bot = LLMChatBot(config, history_manager)

    yield llm_chat_bot

def test_rephase_question(llm_chat_bot) -> None:
    """This function tests rephrase question function."""

    # Test case 1
    chat_history = None
    
    question = 'How much does it cost?'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.debug(f'Rephrased question 1: {rephrase_question}')

    assert rephrase_question == question

    # Test case 2
    chat_history = """Excuse me, which bus goes to the airport?
    The 757 goes directly from the city center to the airport.

    When will it arrive?
    It will arrive in 5 minutes.
    """
    
    question = 'How much does it cost?'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.debug(f'Rephrased question 2: {rephrase_question}')

    assert rephrase_question != question

    # Test case 3
    question = 'What is your name?'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.debug(f'Rephrased question 3: {rephrase_question}')

    assert f"{rephrase_question}" == question

def test_concantenate_documents(llm_chat_bot) -> None:
    """This function tests concantenate documents function."""
    
    pass


def test_get_semantic_answer(llm_chat_bot) -> None:
    """This function tests get semantic answer function."""

    # Test case 1
    llm_chat_bot.indexer.add_document('samples/A.txt', index_name = None)
    llm_chat_bot.indexer.add_document('samples/B.txt', index_name = None)
    llm_chat_bot.indexer.add_document('samples/C.txt', index_name = None)

    question = 'What is the capital of France?'

    answer = llm_chat_bot.get_semantic_answer(
                question, 
                session_id = 'test_session_id', 
                index_name = None, 
                condense_question = True)

    logging.info(f'Answer 1: {answer}')

    # Test case 2
    question = "Who is a better writer, Elon Musk or Shakespeare?"

    answer = llm_chat_bot.get_semantic_answer(
                question, 
                session_id = 'test_session_id', 
                index_name = None, 
                condense_question = True)

    logging.info(f'Answer 2: {answer}')