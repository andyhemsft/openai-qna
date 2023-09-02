import os

from app.utils.config import Config

class MetaData:
    """This class stores the meta data for a document."""

    def __init__(self, config: Config):
        """
        Initialize the MetaData.

        Args:
            config: the config object
        """

        self.config = config
