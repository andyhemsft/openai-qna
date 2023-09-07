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
            user_id=None, 
            is_bot=False
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
    