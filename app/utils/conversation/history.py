import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.config import Config
from app.utils.llm import LLMHelper
from app.utils.conversation import Message
from app.utils.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

class HistoryManager:
    """This class represents a History Manager."""

    def __init__(self, config: Config):
        """
        Initialize the History Manager.

        Args:
            config: the config object
        """

        self.config = config

        # There could be a better way to do this, but for now we will use the same vector store
        self.short_term_store = get_vector_store(config)

        if config.VECTOR_STORE_TYPE == 'faiss':
            # TODO: We may save and load the chat history from local file
            # Currently, the chat history is saved in memory only
            pass

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
            'received_timestamp': message.received_timestamp,
            'responded_timestamp': message.responded_timestamp,
            'user_id': message.user_id,
            'is_bot': message.is_bot
        }

        self.short_term_store.add_texts([text], [metadata])

        # TODO: We need to also log the history in log anlytics for long term storage
    
    def add_qa_pair(self, question: Message, answer: Message) -> None:
        """
        Add a QA pair to the history.

        Args:
            question: the question
            answer: the answer
        Returns:
            none
        """
        
        text = f"Human:{question.text}\nBot:{answer.text}"

        metadata = {
            'session_id': question.session_id,
            'sequence_num': answer.sequence_num,
            'received_timestamp': question.received_timestamp,
            'responded_timestamp': answer.responded_timestamp,
            'user_id': question.user_id,
            'is_bot': answer.is_bot
        }

        self.short_term_store.add_texts([text], [metadata])

    def get_k_most_related_messages(
            self, 
            query: str, 
            session_id: str, 
            k: int = 4,
            score_threshold: float = Config.CHAT_HISTORY_SIMILARITY_THRESHOLD, 
        ) -> List[Message]:
        """
        Get the k most related messages.

        Args:
            query: the query
            session_id: the session id
            k: the number of messages
        Returns:
            the messages
        """

        documents = self.short_term_store.similarity_search(query, 
                                                            k, 
                                                            filter={'session_id': session_id})

        messages = []
        for doc in documents:
            logger.info(f"doc: {doc}")
            if doc[1] < score_threshold:
                continue

            message = self.from_doc_to_message(doc)
            if message.sequence_num > 0: # The 0 sequence number is a placeholder
                messages.append((message.sequence_num, message))

        messages.sort(key=lambda x: x[0])
        
        # TODO: We need to also log the matches in log anlytics for performance investigation

        return [message[1] for message in messages]
    
    def get_k_most_recent_messages(
            self, 
            session_id: str, 
            k: int = 4
        ) -> List[Message]:
        """
        Get the k most recent messages. 
        By default, the messages are sorted in time order.

        Args:
            session_id: the session id
            k: the number of messages
        Returns:
            the messages
        """

        # The vector store cannot be filtered by session_id only, 
        # so we need to get all the messages sorted by sequence_num and return the top k
        documents = self.short_term_store.similarity_search(
            query='',
            k=self.config.CHATBOT_MAX_MESSAGES, 
            filter={'session_id': session_id})

        messages = []
        for doc in documents:
            message = self.from_doc_to_message(doc)
            if message.sequence_num > 0: # The 0 sequence number is a placeholder
                messages.append((message.sequence_num, message))

        messages.sort(key=lambda x: x[0], reverse=True)

        # TODO: We need to also log the matches in log anlytics for performance investigation

        return [message[1] for message in reversed(messages[:k])]
    
    def get_all_messages(self, session_id: str) -> List[Message]:
        """
        Get all the messages.
        By default, the messages are sorted in time order.

        Args:
            session_id: the session id
        Returns:
            the messages
        """

        messages = self.get_k_most_recent_messages(session_id, self.config.CHATBOT_MAX_MESSAGES)

        return messages

    def get_max_sequence_num_and_earliest_time(self, session_id: str) -> int:
        """Get the max sequence number and first timestamp.
        
        Args:
            session_id: the session id
        Returns:
            the max sequence number and first timestamp
        """

        messages = self.get_all_messages(session_id)

        if len(messages) == 0:
            return 0, datetime.now()
        else:
            return messages[-1].sequence_num, messages[0].received_timestamp

    def from_doc_to_message(self, doc) -> Message:
        """
        Convert a document to a message.

        Args:
            doc: the document
        Returns:
            the message
        """

        message = Message(
            text=doc[0].page_content,
            session_id=doc[0].metadata['session_id'],
            sequence_num=doc[0].metadata['sequence_num'],
            received_timestamp=doc[0].metadata['received_timestamp'],
            responded_timestamp=doc[0].metadata['responded_timestamp'],
            user_id=doc[0].metadata['user_id'],
            is_bot=doc[0].metadata['is_bot']
        )

        return message