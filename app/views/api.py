import os, logging 

# Flask modules
from flask import request, jsonify
# App modules
from app import app

from app.config import Config
from app.utils.index import get_indexer
from app.utils.conversation.bot import LLMChatBot
from app.utils.conversation import Message
from app.utils.file.parser import get_parser
from app.config import Config

logger = logging.getLogger(__name__)

# Initialize the LLM Chat Bot
llm_chat_bot = LLMChatBot(Config())

def API_PREFIX(url):
    """Add the API prefix to the url"""

    return '/api' + url

@app.route(API_PREFIX(''))
def hello_api():
    """Print out hello world"""

    return 'Hello World!'

@app.route(API_PREFIX('/parser/document'), methods=['POST'])
def parser_document():
    """Handle document parsing"""

    if request.method == 'POST':
        parser = get_parser(request.json['document_type'])
        text = parser.analyze_read(request.json['source_url'])
        parser.write(request.json['dest_url'], text)

        return jsonify({'data': 'Document parsed'})
    else:
        raise ValueError('Method not supported')

@app.route(API_PREFIX('/indexer'), methods=['POST', 'DELETE'])
def indexer():
    """Handle index creation and deletion"""
    
    config = Config()
    indexer = get_indexer(config)

    if request.method == 'POST':
        # Create index
        indexer.create_index()
        return 'Index created'

    elif request.method == 'DELETE':
        # Delete index
        indexer.delete_index()
        return 'Index deleted'
    
    else:
        raise ValueError('Method not supported')
    
@app.route(API_PREFIX('/indexer/document'), methods=['POST', 'DELETE'])
def document():
    """
        Handle document indexing
    """

    config = Config()
    indexer = get_indexer(config)

    if request.method == 'POST':
        # Index document

        source_url = request.json['source_url']
        index_name = request.json['index_name']

        indexer.add_document(source_url=source_url, index_name=index_name)
        
        return jsonify({'data': 'Document indexed'})

    elif request.method == 'DELETE':
        # Delete document
        
        return jsonify({'data': 'Document delted'})
    
    else:
        raise ValueError('Method not supported')
    

@app.route(API_PREFIX('/indexer/query'), methods=['GET'])
def query():
    """
        Handle query
    """

    config = Config()
    indexer = get_indexer(config)

    if request.method == 'GET':
        # Query index
        
        return 'Query result'

    else:
        raise ValueError('Method not supported')

@app.route(API_PREFIX('/chat/session'), methods=['POST'])
def chat_session():
    """
        Handle chat session
    """

    if request.method == 'POST':
        # Create chat session
        initial_message = llm_chat_bot.initialize_session(user_meta=request.json)
        return jsonify({'session': initial_message.to_json()})

    else:
        raise ValueError('Method not supported')

@app.route(API_PREFIX('/chat/answer'), methods=['POST'])
def chat_answer():
    """
        Handle chat answer
    """

    if request.method == 'POST':
        # Answer question

        message = Message(
            text=request.json['question'],
            session_id=request.json['session_id'],
            user_id=request.json['user_id'],
            is_bot=False # This is a user message
        )

        # Whether to condense the question to the a standalone question
        # based on the chat history
        condense_question = False
        if 'condense_question' in request.json:
            condense_question = request.json['condense_question']

        conversation = 'custom'
        if 'conversation' in request.json:
            conversation = request.json['conversation']

        answer = llm_chat_bot.get_semantic_answer(
            message=message,
            index_name=request.json['index_name'],
            condense_question=condense_question,
            conversation=conversation
        ).message
        
        return jsonify({'answer': answer.to_json()})

    else:
        raise ValueError('Method not supported')
    
@app.route(API_PREFIX('/chat/history'), methods=['GET'])
def chat_history():
    """
        Handle chat history
    """

    if request.method == 'GET':
        # Return chat history
        
        history = llm_chat_bot.get_all_chat_history(request.json['session_id'])
        return jsonify({'history': [chat.to_json() for chat in history]})

    else:
        raise ValueError('Method not supported')