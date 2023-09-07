import logging
import os

from app.config import Config
from app.utils.file.parser import get_parser
from app.utils.index import get_indexer

logger = logging.getLogger(__name__)

def test_hellow_world(client):
    """Test hello world"""

    response = client.get("/ui")
    assert b"Hello World!" in response.data

    response = client.get("/api")
    assert b"Hello World!" in response.data

def test_api_session(client):
    """Test API session"""

    response = client.post("/api/chat/session", json={'user_id': 'test_user_id'})

    logger.info(response.json)
    assert "session" in response.json

def test_api_chat(client):
    """Test API chat"""

    # Test case 1
    test_json = {'user_id': 'test_user_id'}
    response = client.post("/api/chat/session", json=test_json)

    session_id = response.json['session']['session_id']

    test_json = {'session_id': session_id, 
                 'question': 'Hello!', 
                 'user_id': 'test_user_id',
                 'index_name': 'test_index'}
    response = client.post("/api/chat/answer", json=test_json)

    logger.info(response.json)
    assert "answer" in response.json

    # Test case 2
    # Let us index a document first
    indexer = get_indexer(Config())
    index_name = 'test_index'

    documents = ['samples/A.txt', 'samples/B.txt', 'samples/C.txt']

    for doc in documents:
        indexer.add_document(doc, index_name=index_name)

    test_json = {
                    'session_id': session_id,
                    'question': 'Do you know who is Michael Jordan?',
                    'user_id': 'test_user_id',
                    'index_name': index_name
                }
    
    response = client.post("/api/chat/answer", json=test_json)

    logger.info(response.json)
    assert "answer" in response.json

    # Remove the test index
    config = Config()
    indexer = get_indexer(config)
    indexer.drop_all_indexes()

def test_api_parse_document(client):
    """Test API parse document"""

    sample_source_url = 'samples/gpt4_technical_report.pdf'
    sample_dest_url = 'tests/unit/webapp/gpt4_technical_report.txt'

    test_json = {'source_url': sample_source_url, 'dest_url': sample_dest_url, 'document_type': 'pdf'}
    response = client.post("/api/parser/document", json=test_json)

    logger.info(response.json)
    assert "data" in response.json

    # Remove the test output file
    os.remove(sample_dest_url)

def test_api_get_chat_history(client):
    """Test API get chat history"""

    test_json = {'user_id': 'test_user_id'}
    response = client.post("/api/chat/session", json=test_json)

    session_id = response.json['session']['session_id']

    test_json = {'session_id': session_id}
    response = client.get("/api/chat/history", json=test_json)

    logger.info(response.json)
    assert "history" in response.json

def test_api_index_document(client):
    """Test API index document"""

    parser = get_parser('pdf')

    sample_source_url = 'samples/gpt4_technical_report.pdf'
    sample_dest_url = 'tests/unit/webapp/gpt4_technical_report.txt'

    # Prepare test data
    text = parser.analyze_read(sample_source_url)
    parser.write(sample_dest_url, text)

    index_name = 'test_index'

    test_json = {'source_url': sample_dest_url, 'index_name': index_name}
    response = client.post("/api/indexer/document", json=test_json)

    logger.info(response.json)
    
    assert "data" in response.json

    # Remove the test output file
    os.remove(sample_dest_url)

    # Remove the test index
    config = Config()
    indexer = get_indexer(config)
    indexer.drop_all_indexes()
