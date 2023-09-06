import logging
import pytest

from app.config import Config
from app.utils.conversation.bot import LLMChatBot
from app.utils.conversation import Message

logger = logging.getLogger(__name__)


@pytest.fixture()
def llm_chat_bot():
    """This function returns the LLM Chat Bot."""

    config = Config()
    llm_chat_bot = LLMChatBot(config)

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

    # Test case 4
    question = 'Hello'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question,
        chat_history = None)
    
    logger.debug(f'Rephrased question 4: {rephrase_question}')

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

    llm_chat_bot.initialize_session(user_meta = {'user_id': 'test_user_id'})

    message = Message(
        text='What is the capital of France?',
        session_id='test_session_id',
        sequence_num=None,
        timestamp=None,
        user_id=None,
        is_bot=False
    )

    answer = llm_chat_bot.get_semantic_answer(
                message,
                index_name = None, 
                condense_question = True)

    logging.info(f'Answer 1: {answer}')

    # Test case 2
    message = Message(
        text="Who is a better writer, Elon Musk or Shakespeare?",
        session_id='test_session_id',
        sequence_num=None,
        timestamp=None,
        user_id=None,
        is_bot=False
    )

    answer = llm_chat_bot.get_semantic_answer(
                message, 
                index_name = None, 
                condense_question = True)

    logging.info(f'Answer 2: {answer}')

    # Test case 3
    message = Message(
        text="Hello",
        session_id='test_session_id',
        sequence_num=None,
        timestamp=None,
        user_id=None,
        is_bot=False
    )

    answer = llm_chat_bot.get_semantic_answer(
                message,
                index_name = None,
                condense_question = True)
    
    logging.info(f'Answer 3: {answer}')

    # Test case 4
    message = Message(
        text="你好",
        session_id='test_session_id',
        sequence_num=None,
        timestamp=None,
        user_id=None,
        is_bot=False
    )

    answer = llm_chat_bot.get_semantic_answer(
                message,
                index_name = None,
                condense_question = True)
    
    logging.info(f'Answer 3: {answer}')