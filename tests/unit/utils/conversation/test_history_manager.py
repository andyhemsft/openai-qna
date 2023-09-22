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
        received_timestamp="2021-01-01 00:00:00",
        responded_timestamp="2021-01-01 00:00:00",
        user_id="1",
        is_bot=False
    ),

    Message(
        text="Hello, User!",
        session_id="1",
        sequence_num=2,
        received_timestamp="2021-01-01 00:00:01",
        responded_timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="What can you do?",
        session_id="1",
        sequence_num=3,
        received_timestamp="2021-01-01 00:00:02",
        responded_timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=False
    ),

    Message( 
        text="I can answer your questions.",
        session_id="1",
        sequence_num=4,
        received_timestamp="2021-01-01 00:00:03",
        responded_timestamp="2021-01-01 00:00:03",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="Hello, Chatbot!",
        session_id="2",
        sequence_num=1,
        received_timestamp="2021-01-01 00:00:00",
        responded_timestamp="2021-01-01 00:00:00",
        user_id="1",
        is_bot=False
    ),

    Message(
        text="Hello, Alice!",
        session_id="2",
        sequence_num=2,
        received_timestamp="2021-01-01 00:00:01",
        responded_timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=True
    ),

    Message(
        text="What can you do?",
        session_id="2",
        sequence_num=3,
        received_timestamp="2021-01-01 00:00:02",
        responded_timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=False
    ),

    Message( 
        text="I can answer your questions.",
        session_id="2",
        sequence_num=4,
        received_timestamp="2021-01-01 00:00:03",
        responded_timestamp="2021-01-01 00:00:03",
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

def test_add_qa_pair(history_manager):
    """This function tests add QA pair function."""

    for i in range(0, len(test_messages), 2):
        history_manager.add_qa_pair(test_messages[i], test_messages[i+1])


def test_get_k_most_related_messages(history_manager):
    """This function tests get k most related messages function."""

    for i in range(0, len(test_messages), 2):
        history_manager.add_qa_pair(test_messages[i], test_messages[i+1])

    query = "What is your function?"

    messsages = history_manager.get_k_most_related_messages(query=query, session_id="1", k=2)
    
    assert len(messsages) == 2
    assert messsages[0].text == "Human:What can you do?\nBot:I can answer your questions." or \
        messsages[1].text == "Human:What can you do?\nBot:I can answer your questions."

def test_get_k_most_recent_messages(history_manager):
    """This function tests get k most recent messages function."""

    for i in range(0, len(test_messages), 2):
        history_manager.add_qa_pair(test_messages[i], test_messages[i+1])
    
    messsages = history_manager.get_k_most_recent_messages("1", 2)

    assert len(messsages) == 2
    assert messsages[0].text == "Human:Hello, Chatbot!\nBot:Hello, User!"
    assert messsages[1].text == "Human:What can you do?\nBot:I can answer your questions."

def test_get_all_messages(history_manager):
    """This function tests get all messages function."""

    for message in test_messages:
        history_manager.add_message(message)

    messsages = history_manager.get_all_messages("2")

    assert len(messsages) == 4
    assert messsages[0].text == "Hello, Chatbot!"
    assert messsages[1].text == "Hello, Alice!"
    assert messsages[2].text == "What can you do?"
    assert messsages[3].text == "I can answer your questions."