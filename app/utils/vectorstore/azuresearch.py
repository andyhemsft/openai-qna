import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import json
import uuid
import hashlib

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.azuresearch import AzureSearch as AzureSearch_TYPE
from langchain.vectorstores.base import VectorStoreRetriever

from app.utils.vectorstore.base import BaseVectorStore
from app.config import Config

from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,  
)  

logger = logging.getLogger(__name__)

FIELDS_ID = "id"
FIELDS_CONTENT = "content"
FIELDS_CONTENT_VECTOR = "content_vector"

MAX_UPLOAD_BATCH_SIZE = 1000

class AzureSearch(BaseVectorStore):

    def __init__(self, config: Config, embeddings: Embeddings):
        """This function initializes the Azure Search Vector Store."""

        super().__init__(config, embeddings)

        self.azure_search_endpoint = config.AZURE_SEARCH_ENDPOINT
        self.azure_search_api_key = config.AZURE_SEARCH_API_KEY
        self.credential = AzureKeyCredential(self.azure_search_api_key)

        self.search_index_client = SearchIndexClient(endpoint=self.azure_search_endpoint, credential=self.credential)

    def _get_native_azuresearch_client(self, index_name: str) -> SearchClient:
        # Create a client
        
        client = SearchClient(endpoint=self.azure_search_endpoint,
                            index_name=index_name,
                            credential=self.credential)
        
        return client

    def _get_langchain_azuresearch(self, index_name: str) -> AzureSearch_TYPE:
        """This function returns a langchain azuresearch object."""

        embedding_fn = self.embeddings.embed_query
        return AzureSearch_TYPE(
            azure_search_endpoint=self.azure_search_endpoint, 
            azure_search_key=self.azure_search_api_key, 
            index_name=index_name,
            embedding_function=embedding_fn
        )
    
    def _convert_metadata_schema(self, metadata_schema: Dict[str, str]) -> List[SearchField]:
        """This function converts a metadata schema to a list of search fields.
        
        Args:
            metadata_schema: the metadata schema
        Returns:
            a list of search fields
        """

        if metadata_schema is None:
            metadata_schema = {}

        if "metadata" in metadata_schema:
            raise ValueError("metadata is a reserved keyword")

        metadata_schema[FIELDS_ID] = "KEY"
        metadata_schema[FIELDS_CONTENT] = "TEXT"
        metadata_schema[FIELDS_CONTENT_VECTOR] = "VECTOR"

        search_fields = []
        for field_name, field_type in metadata_schema.items():
            if field_type.lower() == "text":
                search_fields.append(SearchableField(name=field_name, type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True))
            elif field_type.lower() == "numeric":
                search_fields.append(SimpleField(name=field_name, type=SearchFieldDataType.Double, facetable=True, filterable=True, sortable=True))
            elif field_type.lower() == "vector":
                search_fields.append(SearchField(name=field_name, type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                                                     searchable=True, vector_search_dimensions=self.config.OPENAI_EMBEDDING_SIZE, 
                                                     vector_search_configuration="my-vector-config"))
            elif field_type.lower() == "key":
                search_fields.append(SimpleField(name=field_name, type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True))
            else:
                raise ValueError(f"Invalid field type: {field_type}")
        
        return search_fields

    def check_existing_index(self, index_name: str = None) -> bool:
        """This function checks if the index exists.
        
        Args:
            index_name: the index name
        Returns:

        """
        return any(index == index_name for index in self.search_index_client.list_index_names())

    def create_index(self, 
                     index_name: str, 
                     metadata_schema: Dict[str, str]=None, 
                     distance_metric: Optional[str]="COSINE"
                     ) -> None:
        """This function creates an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """ 
        if self.check_existing_index(index_name):
            raise ValueError(f"Index {index_name} already exists")

        index_client = SearchIndexClient(
            endpoint=self.azure_search_endpoint, credential=self.credential)
        
        fields = self._convert_metadata_schema(metadata_schema)

        vector_search = VectorSearch(
            algorithm_configurations=[
                HnswVectorSearchAlgorithmConfiguration(
                    name="my-vector-config",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )

        # TODO: Create the semantic settings with the configuration
        semantic_settings = None

        # Create the search index with the semantic settings and vector search
        index = SearchIndex(name=index_name, fields=fields,
                            vector_search=vector_search, semantic_settings=semantic_settings)
        result = index_client.create_or_update_index(index)

        logger.info(f"Index {result.name} created")
    
    def drop_index(self, 
                     index_name: str
                     ) -> None:
        """This function drop an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """
        if not self.check_existing_index(index_name):
            raise ValueError(f"Index {index_name} does not exist")
        
        self.search_index_client.delete_index(index_name)

        logger.info(f"Index {index_name} deleted")

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

        # keys = []
        # if 'keys' in kwargs:
        #     keys = kwargs['keys']
        # else:
        #     for document in documents:
        #         key = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        #         keys.append(f"{index_name}:{key}")

        if not self.check_existing_index(index_name):
            raise ValueError(f"Index {index_name} does not exist")

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]

        self.add_texts(texts, metadatas=metadatas, index_name=index_name)

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

        if not self.check_existing_index(index_name):
            raise ValueError(f"Index {index_name} does not exist")

        client = self._get_native_azuresearch_client(index_name)

        keys = kwargs.get("keys")
        keys = list(map(lambda x: x.replace(':','_'), keys)) if keys else None
        ids = []
        # Write data to index
        data = []
        for i, text in enumerate(texts):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            metadata = metadatas[i] if metadatas else {}
            # Add data to index

            record = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(
                    self.embeddings.embed_query(text), dtype=np.float32
                ).tolist(),
            }

            for k, v in metadata.items():
                record[k] = v

            data.append(record)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                # Reset data
                data = []
        # Upload data to index
        response = client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all([r.succeeded for r in response]):
            return None
        else:
            raise Exception(response)

    def _get_index_fields(self, index_name: str):
        
        index = self.search_index_client.get_index(index_name)
        return [field.name for field in index.fields]

    def _construct_filter_query(self, filter: Dict[str, Any]) -> str:
        """This function constructs the filter query.
        
        Args:
            filter: the filter
        Returns:
            the filter query
        """

        filter_query = []

        for key, value in filter.items():
            if isinstance(value, str):
                filter_query.append(f"{key} eq '{value}'")
            elif isinstance(value, int):
                filter_query.append(f"{key} eq {value}")
            elif isinstance(value, float):
                filter_query.append(f"{key} eq {value}")
            else:
                raise ValueError(f"Invalid filter value: {value}")

        return " and ".join(filter_query)

    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None,
            **kwargs: Any
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """

        # Get langchain azuresearch object
        search_client = self._get_native_azuresearch_client(index_name)
        

        # Perform similarity search

        all_fields = self._get_index_fields(index_name)

        selection = []

        for field in all_fields:
            if field not in ["content_vector"]:
                selection.append(field)
        

        filter_query = self._construct_filter_query(filter) if filter else None
        docs_with_scores: List[Tuple[Document, float]] = []

        if "search_text" in kwargs:
            logger.info(f"Search text: {kwargs['search_text']}")
            # scale_factor = 2
            vector = Vector(value=self.embeddings.embed_query(query), k=k, fields=FIELDS_CONTENT_VECTOR)
            results = search_client.search(  
                search_text=kwargs["search_text"], 
                search_fields=['content'],
                search_mode='any',
                filter=filter_query, 
                # vectors= [vector],
                select=selection,
                top=k
            )
        else:
            vector = Vector(value=self.embeddings.embed_query(query), k=k, fields=FIELDS_CONTENT_VECTOR)
            results = search_client.search(  
                search_text=None, 
                filter=filter_query, 
                vectors= [vector],
                select=selection,
            )  

        for result in results:
            metadata = {}

            if index_name == "embeddings":
                metadata['source'] = result['title']
            else:
                for field in selection:
                    if field not in ["content"]:
                        metadata[field] = result[field]

            doc = Document(
                page_content=result["content"],
                metadata=metadata,
            )
            docs_with_scores.append((doc, result["@search.score"]))

        return docs_with_scores
