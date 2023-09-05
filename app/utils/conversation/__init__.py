import logging
from datetime import datetime

class Message:
    """This class represents a Message."""

    def __init__(
            self, 
            text: str, 
            session_id: str, 
            sequence_num: int, 
            timestamp: datetime, 
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
        self.timestamp = timestamp
        self.user_id = user_id
        self.is_bot = is_bot


    