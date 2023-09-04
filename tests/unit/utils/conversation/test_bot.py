import logging
import pytest

from app.config import Config
from app.utils.conversation.bot import LLMChatBot

logger = logging.getLogger(__name__)


@pytest.fixture()
def llm_chat_bot():
    config = Config()
    llm_chat_bot = LLMChatBot(config, None)

    yield llm_chat_bot

def test_rephase_question(llm_chat_bot) -> None:
    """This function tests rephrase question function."""

    chat_history = """Human: Excuse me, which bus goes to the airport?
    Bot: The 757 goes directly from the city center to the airport.
    Human: When will it arrive?
    Bot: It will arrive in 5 minutes.
    """
    
    question = 'Human: How much does it cost?'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.debug(f'Rephrased question 1: {rephrase_question.content}')

    assert rephrase_question.content != question

    question = 'Human: What is your name?'

    rephrase_question = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.debug(f'Rephrased question 2: {rephrase_question}')

    assert f"Human: {rephrase_question.content}" == question