import os
import logging
from dotenv import load_dotenv



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config():
    """This class represents the config.
    
    Dont change the default values here. Instead, use the .env file.
    """

    load_dotenv(override=True)

    CSRF_ENABLED = True

    # Chat Bot parameters
    CHATBOT_SESSION_TIMEOUT = os.getenv('SESSION_TIMEOUT', 60 * 60) # 1 hour
    CHATBOT_MAX_MESSAGES = os.getenv('MAX_MESSAGES', 100) # Max number of messages per session including bot messages
	
    # LLM parameters
    LLM_TYPE = os.getenv('LLM_TYPE', 'openai')

    # OpenAI parameters
    OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE', 'azure')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION', '2023-05-15')
    OPENAI_ENGINE = os.getenv('OPENAI_ENGINE')
    OPENAI_ENGINE_LIGHT = os.getenv('OPENAI_ENGINE_LIGHT')
    OPENAI_EMBEDDING_ENGINE = os.getenv('OPENAI_EMBEDDING_ENGINE')
    OPENAI_TEMPERATURE = os.getenv('OPENAI_TEMPERATURE', 0.0) # Low temperature (temperature 0.1) to ensure reproducibility.
    OPENAI_MAX_TOKENS = os.getenv('OPENAI_MAX_TOKENS', 1000)
    # for text-embedding-ada-002 model , you will obtain a high-dimensional array (vector) consisting of 1536 floating-point numbers
    OPENAI_EMBEDDING_SIZE = os.getenv('OPENAI_EMBEDDING_SIZE', 1536) 

    # Search parameters

    

    # Vector Store parameters
    VECTOR_STORE_TYPE = os.getenv('VECTOR_STORE_TYPE', 'faiss') # redis, azuresearch, faiss

    # Faiss parameters
    FAISS_LOCAL_FILE_CHATHISTORY = os.getenv('FAISS_LOCAL_FILE_CHATHISTORY', 'data/faiss/chathistory')
    FAISS_LOCAL_FILE_INDEX = os.getenv('FAISS_LOCAL_FILE_INDEXING', 'data/faiss/index')

    # Redis parameters
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', 6379)
    REDIS_PROTOCOL = os.getenv('REDIS_PROTOCOL', 'redis://')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

    # Azure Search parameters
    AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
    AZURE_SEARCH_API_KEY = os.getenv('AZURE_SEARCH_API_KEY')

    # Parser parameters
    SUPPORTED_DOCUMENT_TYPES = set(['pdf', 'excel'])

    PDF_PARSER_TYPE = os.getenv('PDF_PARSER_TYPE', 'pdfloader') # pdfloader, formrecognizer
    DOCUMENT_DEST_LOCATION = os.getenv('DOCUMENT_DEST_LOCATION', 'local') # local, azure

    FORM_RECOGNIZER_ENDPOINT = os.getenv('FORM_RECOGNIZER_ENDPOINT')
    FORM_RECOGNIZER_KEY = os.getenv('FORM_RECOGNIZER_KEY')


    # Azure Blob Storage parameters
    BLOB_ACCOUNT_NAME = os.getenv('BLOB_ACCOUNT_NAME')
    BLOB_ACCOUNT_KEY = os.getenv('BLOB_ACCOUNT_KEY')
    BLOB_CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME')

    # Chunking stategy parameters

    CHUNKING_STRATEGY = os.getenv('CHUNKING_STRATEGY', 'fixed') # fixed, two-layer
    CHUNKING_STRATEGY_MAX_LENGTH = os.getenv('CHUNKING_STRATEGY_MAX_LENGTH', 2000)
    CHUNKING_STRATEGY_OVERLAP = os.getenv('CHUNKING_STRATEGY_OVERLAP', 400)

    # Similarity threshold parameters
    TOP_K_RELATED_DOCUMENTS = os.getenv('TOP_K_RELATED_DOCUMENTS', 3)
    CHAT_HISTORY_SEARCH_TYPE = os.getenv('CHAT_HISTORY_SEARCH_TYPE', 'most_related') # most_related, most_recent
    CHAT_HISTORY_SIMILARITY_THRESHOLD = os.getenv('CHAT_HISTORY_SIMILARITY_THRESHOLD', 0)
    DOCUMENT_SIMILARITY_THRESHOLD = os.getenv('DOCUMENT_SIMILARITY_THRESHOLD', 0.5)

    # SQL DB parameters
    SQL_CONNECTION_STRING = os.getenv('SQL_CONNECTION_STRING')