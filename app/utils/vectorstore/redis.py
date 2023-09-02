from langchain.vectorstores.redis import Redis as Redis_TYPE
from langchain.embeddings.base import Embeddings

from app.config import Config
from app.utils.vectorstore import BaseVectorStore


class RedisExtended(BaseVectorStore):
    """This class represents a Redis Vector Store."""

    def __init__(self, config: Config, embeddings: Embeddings):
        """
        Initialize the Redis Vector Store.

        Args:
            config: the config object
            embeddings: the embeddings model
        """

        super().__init__(config, embeddings)

        