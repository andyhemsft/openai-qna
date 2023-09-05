import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config():
    """This class represents the config.
    
    Dont change the default values here. Instead, use the .env file.
    """

    CSRF_ENABLED = True
	
    # LLM parameters
    LLM_TYPE = os.getenv('LLM_TYPE', 'openai')

    # OpenAI parameters
    OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE', 'azure')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION', '2023-05-15')
    OPENAI_ENGINE = os.getenv('OPENAI_ENGINE')
    OPENAI_EMBEDDING_ENGINE = os.getenv('OPENAI_EMBEDDING_ENGINE')
    OPENAI_TEMPERATURE = os.getenv('OPENAI_TEMPERATURE', 0.7)
    OPENAI_MAX_TOKENS = os.getenv('OPENAI_MAX_TOKENS', 1000)

    # Vector Store parameters
    VECTOR_STORE_TYPE = os.getenv('VECTOR_STORE_TYPE', 'faiss') # redis, azure, faiss

    # Redis parameters
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', 6379)
    REDIS_PROTOCOL = os.getenv('REDIS_PROTOCOL', 'redis://')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

    # Parser parameters
    PDF_PARSER_TYPE = os.getenv('PDF_PARSER_TYPE', 'pdfloader') # pdfloader, formrecognizer
    DOCUMENT_DEST_LOCATION = os.getenv('DOCUMENT_DEST_LOCATION', 'local') # local, azure

    # Azure Blob Storage parameters
    BLOB_ACCOUNT_NAME = os.getenv('BLOB_ACCOUNT_NAME')
    BLOB_ACCOUNT_KEY = os.getenv('BLOB_ACCOUNT_KEY')
    BLOB_CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME')

    # Chunking stategy parameters

    CHUNKING_STRATEGY = os.getenv('CHUNKING_STRATEGY', 'fixed') # fixed, two-layer
    CHUNKING_STRATEGY_MAX_LENGTH = os.getenv('CHUNKING_STRATEGY_MAX_LENGTH', 2000)
    CHUNKING_STRATEGY_OVERLAP = os.getenv('CHUNKING_STRATEGY_OVERLAP', 400)