from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.redis import Redis as Redis_TYPE
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.redis import RedisText

import logging
import numpy as np
from app.config import Config
from app.utils.vectorstore import BaseVectorStore

import hashlib
from redis.client import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TagField, TextField, NumericField

from typing import Any, List, Optional, Dict, Tuple

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

    def create_index(self, 
                     index_name: str, 
                     metadata_schema: Dict[str, str]=None, 
                     distance_metric: Optional[str]="COSINE"
                     ) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
            distance_metric: the distance metric
        Returns:
            none
        """

        content = TextField(name="content")
        content_vector = VectorField("content_vector",
                    "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": self.config.OPENAI_EMBEDDING_SIZE,
                        "DISTANCE_METRIC": distance_metric,
                        "INITIAL_CAP": 1000,
                    })
        
        fields = [content, content_vector]

        if metadata_schema is not None:
            for key, value in metadata_schema.items():
                if value.lower() == 'text':
                    fields.append(TextField(name=key))
                elif value.lower() == 'numeric':
                    fields.append(NumericField(name=key))
                else:
                    raise ValueError(f"Metadata schema value '{value}' is not supported.")
        
        self.redis_client.ft(format_index_name(index_name)).create_index(
            fields = fields,
            # TODO: Langchain use this as the default prefix
            # definition = IndexDefinition(prefix=["doc"], index_type=IndexType.HASH)
            definition = IndexDefinition(prefix=[format_index_name(index_name)], index_type=IndexType.HASH)
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

    def _get_redis_schema(self, index_name: str = None) -> Dict[str, str]:
        """This function gets the redis schema.
        
        Args:
            index_name: the index name
        Returns:
            the redis schema
        
        """    

        # Check if the index exists
        if not self.check_existing_index(index_name):
            raise ValueError(f"Index '{index_name}' does not exist.")
        
        index_info = self.redis_client.ft(format_index_name(index_name)).info()
        logger.debug(f"Index info: {index_info}")

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
        
        logger.debug(f"Schema: {schema}")

        return schema

    
    def _get_langchain_redis(self, index_name: str = None) -> Redis_TYPE:
        """This function gets the langchain redis.
        
        Args:
            index_name: the index name
        Returns:  
            the langchain redis
        """
        
        schema = self._get_redis_schema(index_name)

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
        # TODO: using langchain redis add_documents is too buggy, may remove it completely
        # redis_langchain = self._get_langchain_redis(index_name)
        # logger.debug(f"Schemas: {redis_langchain.schema}")

        keys = []
        if 'keys' in kwargs:
            keys = kwargs['keys']
        else:
            for document in documents:
                key = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
                keys.append(f"{format_index_name(index_name)}:{key}")

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]

        self.add_texts(texts, metadatas=metadatas, index_name=index_name, keys=keys)

        # Add documents to the index
        # redis_langchain.add_documents(documents, keys=keys)

    def add_texts(
            self, 
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None, 
            index_name: str = None, 
            **kwargs: Any
        ) -> None:
        """This function adds texts to the vector store.
        
        Args:
            texts: the texts to add
            index_name: the index name
        Returns:
            none
        """

        # TODO: using langchain redis add_texts is too buggy, may remove it completely
        # redis_langchain = self._get_langchain_redis(index_name)

        keys = []
        if 'keys' in kwargs:
            keys = kwargs['keys']
        else:
            for text in texts:
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                keys.append(f"{format_index_name(index_name)}:{key}")

        for key, text, metadata in zip(keys, texts, metadatas):
            
            metadata['content'] = text

            emdeded_text = self.embeddings.embed_query(text)
            emdeded_text = np.array(emdeded_text).astype(np.float32).tobytes()
            metadata['content_vector'] = emdeded_text

            self.redis_client.hset(key, mapping=metadata)

        # Add texts to the index
        # redis_langchain.add_texts(texts, metadatas=metadatas, keys=keys)

    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """
        
        ## It is too hard to use langchain redis similarity search
        ## We decide to use the native redis similarity search instead

        # redis_langchain = self._get_langchain_redis(index_name)

        # logger.debug(f"Schemas: {redis_langchain.schema}")

        # filter_expression = None

        # for key, value in filter.items():            
        #     # TODO: only text equality is supported for now
        #     if filter_expression is None:
        #         filter_expression = RedisText(key) == value
        #     else:
        #         filter_expression &= RedisText(key) == value

        # return redis_langchain.similarity_search_with_relevance_scores(query, k, filter=filter_expression)

        schema = self._get_redis_schema(index_name)
        
        query, query_params = self._contruct_redis_query(question=query, k=k, filter=filter)

        results = self.redis_client.ft(format_index_name(index_name)).search(query, query_params = query_params)

        docs_with_scores: List[Tuple[Document, float]] = []
        for result in results.docs:
            metadata = {}
            metadata = {"id": result.id}
            
            for key in schema.keys():
                # TODO: Dont return the content and content vector, currently it is hard coded
                if key != "content_vector" and key != "content":
                    metadata[key] = result.__getattribute__(key)
            doc = Document(page_content=result.content, metadata=metadata)

            docs_with_scores.append((doc, float(result.score)))

        logger.debug(f"docs_with_scores: {docs_with_scores}")

        return docs_with_scores

    def _contruct_redis_query(self, question: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> Tuple[Query, Dict[str, Any]]:
        """This function constructs a redis query.
        
        Args:
            question: the question
            filter: the filter
        Returns:
            the redis query
        """

        emdeded_question = self.embeddings.embed_query(question)

        if filter is not None:
            filter_expression = []
            for key, value in filter.items():
                filter_expression.append(f"@{key}:{value}")

            filter_expression = " ".join(filter_expression)

            # Langchain use content_vector as the default vector name
            query = (
                Query(f"({filter_expression})=>[KNN {k} @content_vector $vec as score]")
                .sort_by("score")
                .dialect(2)
            )

        else:
            # Langchain use content_vector as the default vector name
            query = (
                Query(f"*=>[KNN {k} @content_vector $vec as score]")
                .sort_by("score")
                .dialect(2)
            )

        query_params = {'vec': np.array(emdeded_question).astype(np.float32).tobytes()}
            
        return query, query_params


    def get_retriever(self, index_name: Optional[str] = None) -> VectorStoreRetriever:
        """This function gets a retriever.
        
        Args:
            index_name: the index name
        Returns:
            the retriever
        """

        redis_langchain = self._get_langchain_redis(index_name)
        return redis_langchain.as_retriever()