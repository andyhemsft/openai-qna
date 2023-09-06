import logging
import pytest

from app.config import Config
from app.utils.conversation.history import HistoryManager
from app.utils.conversation import Message

logger = logging.getLogger(__name__)

test_messages = [
    Message(
        text="Hello, Chatbot!",
        session_id="1",
        sequence_num=1,
        timestamp="2021-01-01 00:00:00",
        user_id="1",
        is_bot=False
    ),

    Message(
        text="Hello, User!",
        session_id="1",
        sequence_num=2,
        timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="What can you do?",
        session_id="1",
        sequence_num=3,
        timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=False
    ),

    Message( 
        text="I can answer your questions.",
        session_id="1",
        sequence_num=4,
        timestamp="2021-01-01 00:00:03",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="Hello, Chatbot!",
        session_id="2",
        sequence_num=1,
        timestamp="2021-01-01 00:00:00",
        user_id="1",
        is_bot=False
    ),

    Message(
        text="Hello, Alice!",
        session_id="2",
        sequence_num=2,
        timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="What can you do?",
        session_id="2",
        sequence_num=3,
        timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=False
    ),

    Message( 
        text="I can answer your questions.",
        session_id="2",
        sequence_num=4,
        timestamp="2021-01-01 00:00:03",
        user_id="1",
        is_bot=True
    )
]

@pytest.fixture()
def history_manager():
    """This function creates a history manager."""

    config = Config()
    history_manager = HistoryManager(config)

    yield history_manager


def test_add_message(history_manager):
    """This function tests add message function."""

    for message in test_messages:
        history_manager.add_message(message)


def test_get_k_most_related_messages(history_manager):
    """This function tests get k most related messages function."""

    for message in test_messages:
        history_manager.add_message(message)

    query = "What is your function?"

    messsages = history_manager.get_k_most_related_messages(query, "1", 2)
    
    assert len(messsages) == 2
    assert messsages[0].text == "What can you do?"

def test_get_k_most_recent_messages(history_manager):
    """This function tests get k most recent messages function."""

    for message in test_messages:
        history_manager.add_message(message)
    
    messsages = history_manager.get_k_most_recent_messages("1", 3)

    assert len(messsages) == 3
    assert messsages[0].text == "I can answer your questions."
    assert messsages[1].text == "What can you do?"
    assert messsages[2].text == "Hello, User!"

def test_get_all_messages(history_manager):
    """This function tests get all messages function."""

    for message in test_messages:
        history_manager.add_message(message)

    messsages = history_manager.get_all_messages("2")

    assert len(messsages) == 4
    assert messsages[0].text == "I can answer your questions."
    assert messsages[1].text == "What can you do?"
    assert messsages[2].text == "Hello, Alice!"
    assert messsages[3].text == "Hello, Chatbot!"