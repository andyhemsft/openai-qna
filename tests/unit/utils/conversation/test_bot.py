import logging
import pytest
import langchain


from app.config import Config
from app.utils.conversation.bot import LLMChatBot
from app.utils.conversation import Message

logger = logging.getLogger(__name__)
langchain.verbose = True

@pytest.fixture()
def llm_chat_bot():
    """This function returns the LLM Chat Bot."""

    config = Config()
    llm_chat_bot = LLMChatBot(config)

    yield llm_chat_bot

def test_rephase_question(llm_chat_bot) -> None:
    """This function tests rephrase question function."""

    # Test case 1
    chat_history = None
    
    question = 'How much does it cost?'

    rephrase_question, keywords = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.info(f'Rephrased question 1: {rephrase_question}')
    logger.info(f'Keywords: {keywords}')

    assert rephrase_question == question

    # Test case 2
    chat_history = """
        Human: Excuse me, which bus goes to the airport?
        Bot: The 757 goes directly from the city center to the airport.
        Human: When will it arrive?
        Bot: It will arrive in 5 minutes.
        """
    
    question = '需要花費多少錢?'

    rephrase_question, keywords = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.info(f'Rephrased question 2: {rephrase_question}')
    logger.info(f'Keywords: {keywords}')

    assert rephrase_question != question

    # Test case 3
    question = 'What is your name?'

    rephrase_question, keywords = llm_chat_bot.rephrase_question(
        question = question, 
        chat_history = chat_history)

    logger.info(f'Rephrased question 3: {rephrase_question}')
    logger.info(f'Keywords: {keywords}')

    assert f"{rephrase_question}" == question

    # Test case 4
    question = 'Hello'

    rephrase_question, keywords = llm_chat_bot.rephrase_question(
        question = question,
        chat_history = None)
    
    logger.info(f'Rephrased question 4: {rephrase_question}')
    logger.info(f'Keywords: {keywords}')

    assert question in f"{rephrase_question}"

def test_concantenate_documents(llm_chat_bot) -> None:
    """This function tests concantenate documents function."""
    
    pass

def test_get_chat_history(llm_chat_bot) -> None:

    pass

# def test_get_semantic_answer(llm_chat_bot) -> None:
#     """This function tests get semantic answer function."""

#     # Test case 1
#     llm_chat_bot.indexer.add_document('samples/A.txt', index_name = None)
#     llm_chat_bot.indexer.add_document('samples/B.txt', index_name = None)
#     llm_chat_bot.indexer.add_document('samples/C.txt', index_name = None)

#     llm_chat_bot.initialize_session(user_meta = {'user_id': 'test_user_id'})

#     message = Message(
#         text='What is the capital of France?',
#         session_id='test_session_id_1_1'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None, 
#                 condense_question = True).message

#     logging.info(f'Answer 1 (custom): {answer.text}')

#     message = Message(
#         text='What is the capital of France?',
#         session_id='test_session_id_1_2'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None, 
#                 condense_question = True,
#                 conversation = 'langchain').message

#     logging.info(f'Answer 1 (langchain): {answer.text}')


#     # Test case 2
#     message = Message(
#         text="Who is MJ?",
#         session_id='test_session_id_2_1'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message, 
#                 index_name = None, 
#                 condense_question = True).message

#     logging.info(f'Answer 2 (custom): {answer.text}')

#     message = Message(
#         text="Who is MJ?",
#         session_id='test_session_id_2_2'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None, 
#                 condense_question = True,
#                 conversation = 'langchain').message

#     logging.info(f'Answer 2 (langchain): {answer.text}')

#     # Test case 3
#     message = Message(
#         text="Hello",
#         session_id='test_session_id_3_1'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None,
#                 condense_question = True).message
    
#     logging.info(f'Answer 3 (custom): {answer.text}')

#     message = Message(
#         text="Hello",
#         session_id='test_session_id_3_2'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None, 
#                 condense_question = True,
#                 conversation = 'langchain').message

#     logging.info(f'Answer 3 (langchain): {answer.text}')

#     # Test case 4
#     message = Message(
#         text="你好",
#         session_id='test_session_id_4_1'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None,
#                 condense_question = True).message
    
#     logging.info(f'Answer 4 (custom): {answer.text}')

#     message = Message(
#         text="你好",
#         session_id='test_session_id_4_2'
#     )
#     answer = llm_chat_bot.get_semantic_answer(
#                 message,
#                 index_name = None, 
#                 condense_question = True,
#                 conversation = 'langchain').message

#     logging.info(f'Answer 4 (langchain): {answer.text}')

# def test_inserts_citation_into_answer(llm_chat_bot) -> None:
#     """This function tests insert citation into answer function."""

#     # Test case 1
#     answer_txt = 'This is a test answer [[https://test/file.txt]]'
#     answer = Message(
#         text=answer_txt,
#         session_id='test_session_id'
#     )

#     answer, source = llm_chat_bot.insert_citations_into_answer(answer)

#     assert '[1]' in answer.text

#     # Test case 2
#     answer_txt = 'This is a test answer [[https://test/file.txt]] [[sample/file.txt]]'

#     answer = Message(
#         text=answer_txt,
#         session_id='test_session_id'
#     )

#     answer, source = llm_chat_bot.insert_citations_into_answer(answer)

#     assert '[1]' in answer.text
#     assert '[2]' in answer.text

