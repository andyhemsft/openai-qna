import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import Config
from app.utils.llm import LLMHelper
from app.utils.conversation import Message
from app.utils.vectorstore import get_vector_store

class HistoryManager:
    """This class represents a History Manager."""

    def __init__(self, config: Config):
        """
        Initialize the History Manager.

        Args:
            config: the config object
        """

        self.config = config
        
        llm_helper = LLMHelper(config)
        embeddings = llm_helper.get_embeddings()

        # Use the same vector store to store the chat history for active sessions
        self.short_term_store = get_vector_store(config, embeddings)

        # TODO: We need to also log the history in log anlytics for long term storage
        self.long_term_store = None

    def add_message(self, message: Message) -> None:
        """
        Add a message to the history.

        Args:
            message: the message
        Returns:
            none
        """

        text = message.text
        metadata = {
            'session_id': message.session_id,
            'sequence_num': message.sequence_num,
            'timestamp': message.timestamp,
            'user_id': message.user_id,
            'is_bot': message.is_bot
        }

        self.short_term_store.add_texts(text, metadata)

        # TODO: We need to also log the history in log anlytics for long term storage

    def get_k_most_related_messages(self, query: str, session_id: str, k: int = 4) -> List[Message]:
        """
        Get the k most related messages.

        Args:
            query: the query
            session_id: the session id
            k: the number of messages
        Returns:
            the messages
        """

        documents = self.short_term_store.similarity_search(query, k, filter={'session_id': session_id})

        messages = []
        for doc in documents:
            message = Message(
                text=doc.page_content,
                session_id=doc.metadata['session_id'],
                sequence_num=doc.metadata['sequence_num'],
                timestamp=doc.metadata['timestamp'],
                user_id=doc.metadata['user_id'],
                is_bot=doc.metadata['is_bot']
            )
            messages.append(message)
        
        return messages
    
    def get_k_most_recent_messages(self, session_id: str, k: int = 4) -> List[Message]:
        """
        Get the k most recent messages.

        Args:
            session_id: the session id
            k: the number of messages
        Returns:
            the messages
        """

        return None
    
    def get_all_messages(self, session_id: str) -> List[str]:
        """
        Get all the messages.

        Args:
            session_id: the session id
        Returns:
            the messages
        """

        return None

# class HistoryManager:
#     """This class represents a Manager of the Conversation History."""

#     def __init__(self, config: Config, history_store: HistoryStore):
#         """
#         Initialize the Manager.

#         Args:
#             config: the config object
#             history_store: the history store
#         """

#         self.config = config
#         self.history_store = history_store

#     def add_message(self, message: Message) -> None:
#         """
#         Add a message to the history.

#         Args:
#             message: the message
#         Returns:
#             none
#         """

#         raise NotImplementedError
    
#     def get_k_most_related_messages(self, query: str, session_id: str, k: int = 4) -> List[str]:
#         """
#         Get the k most related messages.

#         Args:
#             query: the query
#             session_id: the session id
#             k: the number of messages
#         Returns:
#             the messages
#         """

#         return None

#     def get_k_most_recent_messages(self, session_id: str, k: int = 4) -> List[str]:
#         """
#         Get the k most recent messages.

#         Args:
#             session_id: the session id
#             k: the number of messages
#         Returns:
#             the messages
#         """

#         return None
    
#     def get_all_messages(self, session_id: str) -> List[str]:
#         """
#         Get all the messages.

#         Args:
#             session_id: the session id
#         Returns:
#             the messages
#         """

#         return None