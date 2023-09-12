import logging
from typing import List, Dict
import uuid
from datetime import datetime


from app.config import Config
from app.utils.conversation.history import HistoryManager
from app.utils.llm import LLMHelper
from app.utils.conversation.customprompt import *
from app.utils.conversation import Message
from app.utils.index import get_indexer


logger = logging.getLogger(__name__)

class LLMChatBot:

    def __init__(self, config: Config):
        """
        Initialize the LLM Chat Bot.

        Args:
            config: the config object
        """

        self.config = config
        self.history_manager = HistoryManager(config)

        llm_helper = LLMHelper(config)

        self.llm = llm_helper.get_llm()
        self.embeddings = llm_helper.get_embeddings()

        self.indexer = get_indexer(config)

    def initialize_session(self, user_meta: Dict) -> Message:
        """Initialize a session.
        
        """

        session_id = str(uuid.uuid4())

        # Add the initial message
        initial_message = Message(
            text="",
            session_id=session_id,
            sequence_num=0,
            received_timestamp=datetime.now(),
            responded_timestamp=datetime.now(),
            user_id=user_meta['user_id'],
            is_bot=True
        ) 

        self.history_manager.add_message(initial_message)

        return initial_message
    
    def detect_PII(self, question: str, session_id: str) -> bool:
        """Detect PII in the question.
        
        Args:
            question: the question
            session_id: the session id
        """


        raise NotImplementedError
    
    def detect_intent(self, question: str, session_id: str) -> str:
        """Detect the intent of the question.
        
        Args:
            question: the question
            session_id: the session id
        """

        raise NotImplementedError
    
    def standardize_glossary(self, question: str) -> str:
        """Standardize the glossary."""

        # TODO: implement this by replacing the glossary with the standard one
        return question

    def concatenate_chat_history(self, chat_history: List[Message]) -> str:
        """Concatenate the chat history.
        
        Args:
            chat_history: the chat history
        """

        chat_history_concatenated = '\n'.join(["AI:"+chat.text if chat.is_bot else "Human:"+chat.text for chat in chat_history])
        logger.debug(f'Chat history concatenated: {chat_history_concatenated}')

        return chat_history_concatenated
    
    def rephrase_question(self, question: str, chat_history: List[Message]) -> str:
        """Rephrase the question based on the chat history.
        
        Args:
            question: the question
            chat_history: the chat history
        """

        if chat_history is None or len(chat_history) == 0:
            return question
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [SYSTEM_MESSAGE_PROMPT_REPHRASE_Q, HUMAN_MESSAGE_PROMPT_REPHRASE_Q]
        )

        # chat_history_concatenated = '\n'.join([chat.text for chat in chat_history])
        chat_history_concatenated = self.concatenate_chat_history(chat_history)
        logger.debug(f'Chat history concatenated: {chat_history_concatenated}')

        # get a chat completion from the formatted messages
        rephased_question = self.llm(
            chat_prompt.format_prompt(
                question=question,
                chat_history=chat_history_concatenated, 
            ).to_messages()
        ).content

        return rephased_question
        
    def get_semantic_answer(
            self, 
            message: Message, 
            index_name: str = None, 
            condense_question: bool = True
        ) -> Message:
        """
        Get the semantic answer.

        Args:
            question: the question
            session_id: the session id
            index_name: the index name
        Returns:
            the answer
        """
        # TODO: Detect if there are any PII data in the question

        # Timestamp for recieving the question
        received_timestamp = datetime.now()

        max_sequence_num, earliest_time = self.history_manager.get_max_sequence_num_and_earliest_time(message.session_id)

        # Check if the session is expired
        if max_sequence_num >= self.config.CHATBOT_MAX_MESSAGES or \
            (received_timestamp - earliest_time).total_seconds() > self.config.CHATBOT_SESSION_TIMEOUT:
            # Exceed the maximum number of messages
            # Restart the session
            initial_message = self.initialize_session(user_meta={'user_id': message.user_id})

            return initial_message


        # standardize the glossary
        question = self.standardize_glossary(message.text)

        chat_history = self.history_manager.get_k_most_recent_messages(message.session_id, reverse=False)

        # if condense question
        if condense_question:
            
            logger.info("Condensing the question based on the chat history")
            question = self.rephrase_question(question, chat_history)
            logger.debug(f'Condensed question: {question}')
            
            # get related documents
            related_documents = self.indexer.similarity_search(question, index_name=index_name)

            # concatenate the documents
            documents = self.concatenate_documents(related_documents)

            logger.debug(f'Concatenated documents: {documents}')

            chat_prompt = ChatPromptTemplate.from_messages(
                [SYSTEM_MESSAGE_PROMPT_QA_W_HISTORY, HUMAN_MESSAGE_PROMPT_QA_W_HISTORY]
            )

            # get a chat completion from the formatted messages
            answer = self.llm(
                chat_prompt.format_prompt(
                    summary=documents,
                    chat_history=chat_history,
                    question=question, 
                ).to_messages()
            ).content

        # dont condense question
        else:
            logger.info("Don't condense the question")

            # chat_history = '\n'.join([chat.text for chat in chat_history])
            chat_history = self.concatenate_chat_history(chat_history)
            question_with_chat_history = f'{chat_history}\n{question}'

            logger.debug(f'Question with chat history: {question_with_chat_history}')

            # get related documents
            related_documents = self.indexer.similarity_search(question_with_chat_history, index_name=index_name)

            # concatenate the documents
            documents = self.concatenate_documents(related_documents)

            logger.debug(f'Concatenated documents: {documents}')

            chat_prompt = ChatPromptTemplate.from_messages(
                [SYSTEM_MESSAGE_PROMPT_QA_W_HISTORY, HUMAN_MESSAGE_PROMPT_QA_W_HISTORY]
            )

            # get a chat completion from the formatted messages
            answer = self.llm(
                chat_prompt.format_prompt(
                    summary=documents,
                    chat_history=chat_history,
                    question=question, 
                ).to_messages()
            ).content
        
        # Add the question and answer to the history

        question_message = message
        question_message.sequence_num = max_sequence_num + 1
        question_message.received_timestamp = received_timestamp
        question_message.responded_timestamp = datetime.now()
        question_message.is_bot = False

        answer_message = Message(
            text=answer,
            session_id=message.session_id,
            sequence_num=max_sequence_num + 2,
            received_timestamp=received_timestamp,
            responded_timestamp=datetime.now(),
            user_id=message.user_id,
            is_bot=True
        )

        self.history_manager.add_message(question_message)
        self.history_manager.add_message(answer_message)

        return answer_message
    
    def concatenate_documents(self, documents: List[str]) -> str:
        """Concatenate the documents.
        
        Args:
            documents: the documents
        """

        result = ''

        for document in documents:
            result += document[0].page_content + '\n'

        return result
    
    def get_followup_question(self, question: str, session_id: str) -> str:
        """
        Get the follow-up question.

        Args:
            question: the question
        Returns:
            the follow-up question
        """

        raise NotImplementedError
    
    def insert_citations_into_answer(self, answer: str, file_list: List[str], session_id: str):
        """
        Insert citations into the answer.

        Args:
            answer: the answer
        Returns:
            the answer with citations
        """

        raise NotImplementedError
    
    def get_chat_history(self, session_id: str) -> List[Message]:
        """Get the chat history for a session.
        
        Args:
            session_id: the session id
        """

        return self.history_manager.get_all_messages(session_id)
    
