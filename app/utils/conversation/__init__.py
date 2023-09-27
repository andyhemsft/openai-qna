import logging
from datetime import datetime

class Message:
    """This class represents a Message."""

    def __init__(
            self, 
            text: str, 
            session_id: str, 
            sequence_num: int = None, 
            received_timestamp: datetime = None,
            responded_timestamp: datetime = None, 
            user_id: str = None, 
            is_bot: int = 0
        ):
        """
        Initialize the Message.

        Args:
            message: the message
        """

        self.text = text
        self.session_id = session_id
        self.sequence_num = sequence_num
        self.received_timestamp = received_timestamp
        self.responded_timestamp = responded_timestamp
        self.user_id = user_id
        self.is_bot = is_bot

    def to_json(self):
        """Convert the message to json."""

        return {
            'text': self.text,
            'session_id': self.session_id,
            'sequence_num': self.sequence_num,
            'received_timestamp': self.received_timestamp,
            'responded_timestamp': self.responded_timestamp,
            'user_id': self.user_id,
            'is_bot': self.is_bot
        }

class Source:
    """This class represents a Source."""

    def __init__(self):
        pass

class Answer:
    """This class represents an Answer."""

    def __init__(self, message: Message, source: Source):
        """Initialize the Answer.
        
        Args:
            message: the message
            source: the source
        """
        self.message = message
        self.source = source