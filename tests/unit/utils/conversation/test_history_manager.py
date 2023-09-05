import logging
import pytest

from app.config import Config
from app.utils.conversation.history import HistoryManager, HistoryStore

logger = logging.getLogger(__name__)


@pytest.fixture()
def history_manager():
    """This function creates a history manager."""

    config = Config()

    history_store = HistoryStore(config)
    history_manager = HistoryManager(config, history_store)

    yield history_manager


def test_add_message(history_manager):
    """This function tests add message function."""

    pass

def test_get_k_most_related_messages(history_manager):
    """This function tests get k most related messages function."""

    pass

def test_get_k_most_recent_messages(history_manager):
    """This function tests get k most recent messages function."""

    pass

def test_get_all_messages(history_manager):
    """This function tests get all messages function."""

    pass