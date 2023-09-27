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
        is_bot=0
    ),

    Message(
        text="Hello, User!",
        session_id="1",
        sequence_num=2,
        received_timestamp="2021-01-01 00:00:01",
        responded_timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=1
    ),

    Message(
        text="What can you do?",
        session_id="1",
        sequence_num=3,
        received_timestamp="2021-01-01 00:00:02",
        responded_timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=0
    ),

    Message( 
        text="I can answer your questions.",
        session_id="1",
        sequence_num=4,
        received_timestamp="2021-01-01 00:00:03",
        responded_timestamp="2021-01-01 00:00:03",
        user_id="1",
        is_bot=1
    ),

    Message(
        text="Hello, Chatbot!",
        session_id="2",
        sequence_num=1,
        received_timestamp="2021-01-01 00:00:00",
        responded_timestamp="2021-01-01 00:00:00",
        user_id="1",
        is_bot=0
    ),

    Message(
        text="Hello, Alice!",
        session_id="2",
        sequence_num=2,
        received_timestamp="2021-01-01 00:00:01",
        responded_timestamp="2021-01-01 00:00:01",
        user_id="1",
        is_bot=1
    ),

    Message(
        text="What can you do?",
        session_id="2",
        sequence_num=3,
        received_timestamp="2021-01-01 00:00:02",
        responded_timestamp="2021-01-01 00:00:02",
        user_id="1",
        is_bot=0
    ),

    Message( 
        text="I can answer your questions.",
        session_id="2",
        sequence_num=4,
        received_timestamp="2021-01-01 00:00:03",
        responded_timestamp="2021-01-01 00:00:03",
        user_id="1",
        is_bot=1
    )
]

@pytest.fixture()
def history_managers():
    """This function creates a history manager."""

    history_managers = {}

    # Load config
    config = Config()

    # Save old config
    old_vector_store_type = Config.VECTOR_STORE_TYPE

    # Load FAISS vector store
    Config.VECTOR_STORE_TYPE = 'faiss'
    history_managers['faiss'] = HistoryManager(config)

    # Load Redis vector store
    Config.VECTOR_STORE_TYPE = 'redis'
    history_managers['redis'] = HistoryManager(config)

    # Restore old config
    Config.VECTOR_STORE_TYPE = old_vector_store_type

    yield history_managers


def test_add_message(history_managers):
    """This function tests add message function."""

    for vector_store_type, history_manager in history_managers.items():
        logger.info(f'Testing {vector_store_type} add message')
        for message in test_messages:
            history_manager.add_message(message)
        
        history_manager.clear_all_history()

def test_add_qa_pair(history_managers):
    """This function tests add QA pair function."""

    for vector_store_type, history_manager in history_managers.items():
        logger.info(f'Testing {vector_store_type} add QA pair')
        for i in range(0, len(test_messages), 2):
            history_manager.add_qa_pair(test_messages[i], test_messages[i+1])
        
        history_manager.clear_all_history()


def test_get_k_most_related_messages(history_managers):
    """This function tests get k most related messages function."""

    for vector_store_type, history_manager in history_managers.items():
        logger.info(f'Testing {vector_store_type} get k most related messages')
        for i in range(0, len(test_messages), 2):
            history_manager.add_qa_pair(test_messages[i], test_messages[i+1])

        query = "What is your function?"

        messsages = history_manager.get_k_most_related_messages(query=query, session_id="1", k=2)
        
        assert len(messsages) == 2
        assert messsages[0].text == "Human:What can you do?\nBot:I can answer your questions." or \
            messsages[1].text == "Human:What can you do?\nBot:I can answer your questions."
        
        history_manager.clear_all_history()

def test_get_k_most_recent_messages(history_managers):
    """This function tests get k most recent messages function."""

    for vector_store_type, history_manager in history_managers.items():
        logger.info(f'Testing {vector_store_type} get k most recent messages')
        for i in range(0, len(test_messages), 2):
            history_manager.add_qa_pair(test_messages[i], test_messages[i+1])
        
        messsages = history_manager.get_k_most_recent_messages("1", 2)

        assert len(messsages) == 2
        assert messsages[0].text == "Human:Hello, Chatbot!\nBot:Hello, User!"
        assert messsages[1].text == "Human:What can you do?\nBot:I can answer your questions."

        history_manager.clear_all_history()

def test_get_all_messages(history_managers):
    """This function tests get all messages function."""

    for vector_store_type, history_manager in history_managers.items():
        logger.info(f'Testing {vector_store_type} get all messages')
        for message in test_messages:
            history_manager.add_message(message)

        messsages = history_manager.get_all_messages("2")

        assert len(messsages) == 4
        assert messsages[0].text == "Hello, Chatbot!"
        assert messsages[1].text == "Hello, Alice!"
        assert messsages[2].text == "What can you do?"
        assert messsages[3].text == "I can answer your questions."

        history_manager.clear_all_history()