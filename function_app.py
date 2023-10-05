import azure.functions as func
import logging
import json


from app.config import Config
from app.utils.index import get_indexer
from app.utils.conversation.bot import LLMChatBot
from app.utils.conversation import Message
from app.utils.file.parser import get_parser
from app.config import Config

# logger = logging.getLogger(__name__)

app = func.FunctionApp()

# Initialize the LLM Chat Bot
llm_chat_bot = LLMChatBot(Config())

@app.function_name(name="InitSession")
@app.route(route="chat/session")
def chat_session(req: func.HttpRequest) -> func.HttpResponse:
    
    if req.method.upper() == 'POST':
        # Initialize a new session
        initial_message = llm_chat_bot.initialize_session(user_meta=req.get_json())
        
        return func.HttpResponse(json.dumps({'session': initial_message.to_json()})) 
    else:
        raise ValueError('Method not supported')
     
@app.function_name(name="ParseDocument")
@app.route(route="parser/document")
def parser_document(req: func.HttpRequest) -> func.HttpResponse:
    """Handle document parsing"""

    if req.method.upper() != 'POST':
        parser = get_parser(req.get_json()['document_type'])
        text = parser.analyze_read(req.get_json()['source_url'])
        parser.write(req.get_json()['dest_url'], text)

        return func.HttpResponse(json.dumps({'data': 'Document parsed'}))
    else:
        raise ValueError('Method not supported')


@app.function_name(name="Indexer")
@app.route(route="indexer")
def indexer(req: func.HttpRequest) -> func.HttpResponse:
    """Handle index creation and deletion"""
    
    config = Config()
    indexer = get_indexer(config)

    index_name = req.get_json()['index_name']
    if index_name is None:
        return {'data': 'index_name is required'}, 400

    if req.method.upper() == 'POST':
        
        if indexer.check_existing_index(index_name=index_name):
            return func.HttpResponse(json.dumps({'data': 'Index already exists'}), status_code=400)

        # Create index
        indexer.create_index(index_name=index_name)
        return func.HttpResponse('Index created')

    elif req.method.upper() == 'DELETE':
        # Delete index
        indexer.delete_index(index_name=index_name)
        return func.HttpResponse('Index deleted')
    
    else:
        raise ValueError('Method not supported')
    
@app.function_name(name="IndexDocument")
@app.route(route="indexer/document")
def document(req: func.HttpRequest) -> func.HttpResponse:
    """
        Handle document indexing
    """

    config = Config()
    indexer = get_indexer(config)

    if req.method.upper() == 'POST':
        # Index document

        source_url = req.get_json()['source_url']
        index_name = req.get_json()['index_name']

        if not indexer.check_existing_index(index_name=index_name):
            return func.HttpResponse(json.dumps({'data': f'Index {index_name} does not exist'}), status_code=400)

        indexer.add_document(source_url=source_url, index_name=index_name)
        
        return func.HttpResponse(json.dumps({'data': 'Document indexed'}))

    elif req.method.upper == 'DELETE':
        # Delete document
        
        return func.HttpResponse(json.dumps({'data': 'Document deleted'}))
    
    else:
        raise ValueError('Method not supported')
    


@app.function_name(name="ChatAnswer")
@app.route(route="chat/answer")
def chat_answer(req: func.HttpRequest) -> func.HttpResponse:
    """
        Handle chat answer
    """

    if req.method.upper() == 'POST':
        # Answer question

        message = Message(
            text=req.get_json()['question'],
            session_id=req.get_json()['session_id'],
            user_id=req.get_json()['user_id'],
            is_bot=False # This is a user message
        )

        # Whether to condense the question to the a standalone question
        # based on the chat history
        condense_question = False
        if 'condense_question' in req.get_json():
            condense_question = req.get_json()['condense_question']

        conversation = 'custom'
        if 'conversation' in req.get_json():
            conversation = req.get_json()['conversation']

        answer = llm_chat_bot.get_semantic_answer(
            message=message,
            index_name=req.get_json()['index_name'],
            condense_question=condense_question,
            conversation=conversation
        ).message
        
        return func.HttpResponse(json.dumps({'answer': answer.to_json()}))

    else:
        raise ValueError('Method not supported')
    
@app.function_name(name="ChatHistory")
@app.route(route="chat/history")
def chat_history(req: func.HttpRequest) -> func.HttpResponse:
    """
        Handle chat history
    """

    if req.method.upper() == 'GET':
        # Return chat history
        
        history = llm_chat_bot.get_all_chat_history(req.get_json()['session_id'])
        return func.HttpResponse(json.dumps({'history': [chat.to_json() for chat in history]}))

    else:
        raise ValueError('Method not supported')