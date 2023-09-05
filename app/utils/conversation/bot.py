import logging
from typing import List, Dict
import uuid


from app.config import Config
from app.utils.conversation.history import HistoryManager
from app.utils.llm import LLMHelper
from app.utils.conversation.customprompt import *
from app.utils.vectorstore import get_vector_store
from app.utils.index import get_indexer


logger = logging.getLogger(__name__)

class LLMChatBot:

    def __init__(self, config: Config, history_manager: HistoryManager):
        """
        Initialize the LLM Chat Bot.

        Args:
            config: the config object
        """

        self.config = config
        self.history_manager = history_manager

        llm_helper = LLMHelper(config)

        self.llm = llm_helper.get_llm()
        self.embeddings = llm_helper.get_embeddings()

        vector_store = get_vector_store(config, self.embeddings)
        self.indexer = get_indexer(config, vector_store)

    def initialize_session(self, user_meta: Dict) -> str:
        """Initialize a session.
        
        """

        session_id = str(uuid.uuid4())

        return session_id
    
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

    
    def rephrase_question(self, question: str, chat_history: str) -> str:
        """Rephrase the question.
        
        Args:
            question: the question
            chat_history: the chat history
        """

        if chat_history == '' or chat_history is None:
            return question
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [SYSTEM_MESSAGE_PROMPT_REPHRASE_Q, HUMAN_MESSAGE_PROMPT_REPHRASE_Q]
        )

        # get a chat completion from the formatted messages
        rephased_question = self.llm(
            chat_prompt.format_prompt(
                question=question,
                chat_history=chat_history, 
            ).to_messages()
        ).content

        return rephased_question
        
    def get_semantic_answer(
            self, 
            question: str, 
            session_id: str, 
            index_name: str = None, 
            condense_question: bool = True
        ) -> str:
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
        
        # standardize the glossary
        question = self.standardize_glossary(question)

        chat_history = self.history_manager.get_k_most_recent_messages(session_id)

        # if condense question
        if condense_question:
            
            logger.info("Condensing the question based on the chat history")
            question = self.rephrase_question(question, chat_history)
            
            # get related documents
            related_documents = self.indexer.similarity_search(question, index_name=index_name)

            # concatenate the documents
            documents = self.concatenate_documents(related_documents)

            chat_prompt = ChatPromptTemplate.from_messages(
                [SYSTEM_MESSAGE_PROMPT_QA_WO_HISTORY, HUMAN_MESSAGE_PROMPT_QA_WO_HISTORY]
            )

            # get a chat completion from the formatted messages
            answer = self.llm(
                chat_prompt.format_prompt(
                    summary=documents,
                    question=question, 
                ).to_messages()
            ).content

        # dont condense question
        else:
            logger.info("Don't condense the question")
            raise NotImplementedError

        return answer
    
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
    
