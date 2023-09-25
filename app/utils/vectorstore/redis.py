from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.redis import Redis as Redis_TYPE
from langchain.embeddings.base import Embeddings

import logging
from app.config import Config
from app.utils.vectorstore import BaseVectorStore

from redis.client import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField

from typing import Any, List, Optional

logger = logging.getLogger(__name__)

format_index_name = lambda index_name: f"{index_name}"

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
        self.redis_url = f"{config.REDIS_PROTOCOL}:{config.REDIS_PASSWORD}@{config.REDIS_HOST}:{config.REDIS_PORT}"
        self.redis_client = self.get_client(self.redis_url)

    def get_client(self, redis_url: str, **kwargs: Any) -> Redis:
        """Get a redis client from the connection url given. This helper accepts
        urls for Redis server (TCP with/without TLS or UnixSocket) as well as
        Redis Sentinel connections.

        Redis Cluster is not supported.

        Before creating a connection the existence of the database driver is checked
        an and ValueError raised otherwise

        To use, you should have the ``redis`` python package installed.

        Example:
            .. code-block:: python

                redis_client = get_client(
                    redis_url="redis://username:password@localhost:6379"
                    index_name="my-index",
                    embedding_function=embeddings.embed_query,
                )

        To use a redis replication setup with multiple redis server and redis sentinels
        set "redis_url" to "redis+sentinel://" scheme. With this url format a path is
        needed holding the name of the redis service within the sentinels to get the
        correct redis server connection. The default service name is "mymaster". The
        optional second part of the path is the redis db number to connect to.

        An optional username or password is used for booth connections to the rediserver
        and the sentinel, different passwords for server and sentinel are not supported.
        And as another constraint only one sentinel instance can be given:

        Example:
            .. code-block:: python

                redis_client = get_client(
                    redis_url="redis+sentinel://username:password@sentinelhost:26379/mymaster/0"
                    index_name="my-index",
                    embedding_function=embeddings.embed_query,
                )
        """

        # Initialize with necessary components.
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis>=4.1.0`."
            )

        # check if normal redis:// or redis+sentinel:// url
        if redis_url.startswith("redis+sentinel"):
            raise ValueError(
                "Redis Sentinel is not supported by this version of the vector store."
            )
        elif redis_url.startswith("rediss+sentinel"):  # sentinel with TLS support enables
            raise ValueError(
                "Redis Sentinel with TLS is not supported by this version of the vector store."
            )
        else:
            # connect to redis server from url, reconnect with cluster client if needed
            redis_client = redis.from_url(redis_url, **kwargs)
            if self._check_for_cluster(redis_client):
                redis_client.close()
                redis_client = self._redis_cluster_client(redis_url, **kwargs)
        return redis_client
    
    def _check_for_cluster(self, redis_client: Redis) -> bool:
        import redis

        try:
            cluster_info = redis_client.info("cluster")
            return cluster_info["cluster_enabled"] == 1
        except redis.exceptions.RedisError:
            return False


    def _redis_cluster_client(self, redis_url: str, **kwargs: Any) -> Redis:
        """Get a redis cluster client from the connection url given.
        
        Args:
            redis_url: the redis url
            kwargs: the keyword arguments
        Returns:
            the redis cluster client
        """
        from redis.cluster import RedisCluster

        return RedisCluster.from_url(redis_url, **kwargs)

    def create_index(self, index_name: str, distance_metric: Optional[str]="COSINE") -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
            distance_metric: the distance metric
        Returns:
            none
        """

        content = TextField(name="content")
        metadata = TextField(name="metadata")
        content_vector = VectorField("content_vector",
                    "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": 1536,
                        "DISTANCE_METRIC": distance_metric,
                        "INITIAL_CAP": 1000,
                    })
        
        self.redis_client.ft(format_index_name(index_name)).create_index(
            fields = [content, metadata, content_vector],
            # TODO: Langchain use this as the default prefix
            definition = IndexDefinition(prefix=["doc"], index_type=IndexType.HASH)
        )

    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        self.redis_client.ft(format_index_name(index_name)).dropindex(True)

    def check_existing_index(self, index_name: str = None) -> bool:
        """This function checks if the index exists.
        
        Args:
            index_name: the index name
        Returns:

        """

        try:
            self.redis_client.ft(format_index_name(index_name)).info()
            return True
        except:
            return False
        
    def _get_langchain_redis(self, index_name: str = None) -> Redis_TYPE:
        """This function gets the langchain redis.
        
        Args:
            index_name: the index name
        Returns:  
            the langchain redis
        """
        # Check if the index exists
        if not self.check_existing_index(index_name):
            raise ValueError(f"Index '{index_name}' does not exist.")
        
        index_info = self.redis_client.ft(format_index_name(index_name)).info()
        logger.info(f"Index info: {index_info}")

        schema = {}
        # Langchain Redis requires the schema to be defined
        for attribute in index_info['attributes']:
            
            identifier = None
            data_type = None

            for i in range(len(attribute) - 1):
                if attribute[i] == b'identifier':
                    identifier = attribute[i+1].decode("utf-8")
                elif attribute[i] == b'type':
                    data_type = attribute[i+1].decode("utf-8")

            assert identifier is not None
            assert data_type is not None

            schema[identifier] = data_type
        
        logger.info(f"Schema: {schema}")

        # Create langchain Redis from existing index
        redis_langchain = Redis_TYPE.from_existing_index(
            self.embeddings, 
            redis_url=self.redis_url, 
            index_name=format_index_name(index_name),
            schema=schema
        )
        
        return redis_langchain
        
    def add_documents(
            self, 
            documents: List[Document], 
            index_name: Optional[str] = None,
            **kwargs: Any
        ) -> None:

        """This function adds documents to the vector store.

        Args:
            documents: the documents to add
            index_name: the index name

        Returns:
            none
        """

        redis_langchain = self._get_langchain_redis(index_name)

        # Add documents to the index
        redis_langchain.add_documents(documents)

    def get_retriever(self, index_name: str | None = None) -> VectorStoreRetriever:
        """This function gets a retriever.
        
        Args:
            index_name: the index name
        Returns:
            the retriever
        """

        redis_langchain = self._get_langchain_redis(index_name)
        return redis_langchain.as_retriever()