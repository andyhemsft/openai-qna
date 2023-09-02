import logging

from app.utils.config import Config

class HistoryStore:
    """This class represents a History Store."""

    def __init__(self, config: Config):
        """
        Initialize the History Store.

        Args:
            config: the config object
        """

        self.config = config
        self.logger = logging.getLogger(__name__)

class Manager:
    """This class represents a Manager of the Conversation History."""

    def __init__(self, config: Config, history_store: HistoryStore):
        """
        Initialize the Manager.

        Args:
            config: the config object
            history_store: the history store
        """

        self.config = config
        self.history_store = history_store

    def add_message(self, message):
        """
        Add a message to the history.

        Args:
            message: the message
        Returns:
            none
        """

        raise NotImplementedError

    def get_k_most_related_messages(self, message: str, k=None: int):
        """
        Get the k most related messages.

        Args:
            message: the message
            k: the number of messages
        Returns:
            the messages
        """

        raise NotImplementedError

    def get_k_most_recent_messages(self, session_id: int, k=None: int):
        """
        Get the k most recent messages.

        Args:
            k: the number of messages
        Returns:
            the messages
        """

        raise NotImplementedError