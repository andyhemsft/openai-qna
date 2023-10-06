import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.config import Config
from app.utils.file.storage import get_storage_client, BLOB_STORAGE_PATERN

class Message:
    """This class represents a Message."""

    def __init__(
            self, 
            text: str, 
            session_id: str, 
            sequence_num: int = None, 
            received_timestamp: str = None,
            responded_timestamp: str = None, 
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

    def __init__(self, source_urls: Dict[int, str]):
        self.source_urls = source_urls
        self.source_urls = self._transform_url()

    def _transform_url(self) -> Tuple[str, str, str]:
        """This function transforms the url."""

        blob_storage_client = get_storage_client('blob')
        source_urls = []

        # if the source is a blob url, then transform it to a sas url
        for key in self.source_urls:
            
            file_name = self._extract_file_name(self.source_urls[key])
            if BLOB_STORAGE_PATERN in self.source_urls[key]:
                self.source_urls[key] = blob_storage_client.get_sas_url(self.source_urls[key])

            source_urls.append((f'[{key}]', file_name, self.source_urls[key]))
        
        return source_urls

    def _extract_file_name(self, url: str) -> str:
        """This function extracts the file name from the url."""

        return url.split('/')[-1]

    def to_json(self):
        """Convert the source to json."""

        source_urls = {'source_urls': []}
        for source in self.source_urls:
            source_urls['source_urls'].append({
                'index': source[0],
                'file_name': source[1],
                'url': source[2]
            })
        
        return source_urls
    
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