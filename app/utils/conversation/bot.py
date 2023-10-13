import logging
from typing import List, Dict, Tuple
import uuid
from datetime import datetime
import re
import copy

from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from app.config import Config
from app.utils.conversation.history import HistoryManager
from app.utils.llm import LLMHelper
from app.utils.conversation.customprompt import *
from app.utils.conversation import Message, Answer, Source
from app.utils.index import get_indexer

from app.utils.agent.base import AgentExecutor
from app.utils.agent.retrieval import get_doc_retrieval_agent
from app.utils.agent.mdrt import get_mdrt_qna_agent

from app.utils.conversation.intent import LLMIntentDetector
from app.utils.conversation.customprompt import DEFAULT_PROMPT, CHAT_SUMMARIZATION_PROMPT

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
        self.llm_light = llm_helper.get_llm(light_weight=True)
        self.embeddings = llm_helper.get_embeddings()

        self.indexer = get_indexer(config)

        self.llm_intent_detector = LLMIntentDetector(config=self.config)

        self.default_prompt = DEFAULT_PROMPT

    def initialize_session(self, user_meta: Dict) -> Message:
        """Initialize a session.
        
        """

        session_id = str(uuid.uuid4()).replace('-', '')

        # Add the initial message
        initial_message = Message(
            text="",
            session_id=session_id,
            sequence_num=0,
            received_timestamp=str(datetime.now()),
            responded_timestamp=str(datetime.now()),
            user_id=user_meta['user_id'],
            is_bot=1
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
    
    def detect_intent(self, question: str) -> str:
        """Detect the intent of the question.
        
        Args:
            question: the question
        """

        return self.llm_intent_detector.detect_intent(question)
    
    def _summarize_chat_history(self, chat_history) -> str:

        """Summarize the chat history."""

        # get a chat completion from the formatted messages
        summarized_chat = self.llm(
            CHAT_SUMMARIZATION_PROMPT.format_prompt(
                chat=chat_history, 
            ).to_messages()
        ).content

        return summarized_chat

    def standardize_glossary(self, question: str) -> str:
        """Standardize the glossary."""

        # TODO: implement this by replacing the glossary with the standard one
        return question

    def concatenate_chat_history(self, chat_history: List[Message]) -> str:
        """Concatenate the chat history.
        
        Args:
            chat_history: the chat history
        """

        chat_history_concatenated = '\n'.join(chat.text for chat in chat_history)
        # logger.debug(f'Chat history concatenated: {chat_history_concatenated}')

        return chat_history_concatenated
    
    def _parse_rephrased_question(self, rephrased_question: str):
        """Parse the rephrased question."""

        # The rephrased question is in the format of: 
        # Final Answer: rephrased_question [[[keyword1, keyword2, keyword3]]]

        # Locate the keywords using regex first, dont use split
        keywords = re.findall(r'\[\[\[.*?\]\]\]', rephrased_question)

        if len(keywords) > 0:
            keywords = keywords[0].split('[[[')[1].split(']]]')[0].split(',')
            keywords = [keyword.strip() for keyword in keywords]
            keywords = " ".join(keywords)

        # Get the rephrased question
        rephrased_question = rephrased_question.split('Final Answer: ')

        if len(rephrased_question) > 1:
            # Dont contain the keywords
            rephrased_question = rephrased_question[1].split('[[[')[0].strip()

        else:
            # THe Final Answer is not in the output
            rephrased_question = rephrased_question[0].split('[[[')[0].strip()

        return rephrased_question, keywords

    def rephrase_question(self, question: str, chat_history: str):
        """Rephrase the question based on the chat history.
        
        Args:
            question: the question
            chat_history: the chat history
        """

        # if chat_history is None or len(chat_history) == 0:
        #     return question, None
        
        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [SYSTEM_MESSAGE_PROMPT_REPHRASE_Q, HUMAN_MESSAGE_PROMPT_REPHRASE_Q]
        # )

        # Use rephase with keyword prompt
        # chat_prompt = ChatPromptTemplate.from_messages(
        #     [SYSTEM_MESSAGE_PROMPT_REPHRASE_KEYWORD, HUMAN_MESSAGE_PROMPT_REPHRASE_KEYWORD]
        # )

        chat_prompt = REPHASE_QUESTION_PROMPT

        # get a chat completion from the formatted messages
        output = self.llm(
            chat_prompt.format_prompt(
                question=question,
                chat_history=chat_history, 
            ).to_messages()
        ).content

        logger.info(f'Output of rephase question prompt: {output}')

        rephrased_question, keywords = self._parse_rephrased_question(output)

        return rephrased_question, keywords

    def get_chat_history(self, question: str, session_id: str, search_type: str) -> List[Message]:
        """
        Get the chat history.

        Args:
            session_id: the session id
            search_type: the search type
        Returns:
            the chat history
        """

        if search_type == 'most_recent':
            return self.history_manager.get_k_most_recent_messages(session_id=session_id)

        elif search_type == 'most_related':
            return self.history_manager.get_k_most_related_messages(query=question, session_id=session_id)

        else:  
            NotImplementedError

    def _get_semantic_answer_custom(
            self, 
            message: Message, 
            index_name: str = None,
            condense_question: bool = True
        ) -> Answer:
        """
        Get the semantic answer using custom logic.

        Args:
            question: the question
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

        chat_history = self.get_chat_history(question, message.session_id, self.config.CHAT_HISTORY_SEARCH_TYPE)
        chat_history = self.concatenate_chat_history(chat_history)

        keywords = None

        if condense_question:
            # Concatenate the chat history
            logger.info("Condesing question")
            question_with_chat_history, keywords = self.rephrase_question(question, chat_history)

        else:
            logger.info("Dont condese question")
            
            # Summarize the chat history
            # chat_history = self._summarize_chat_history(chat_history)

            question_with_chat_history = f'{chat_history}\n{question}'

        logger.info(f'Question with chat history: {question_with_chat_history}')
        logger.info(f'Keywords: {keywords}')
        
        intent = self.detect_intent(question_with_chat_history)

        if intent == "DocRetrieval":
            # get related documents
            logger.info(f"indices: {index_name}")

            def isEnglish(s):
                try:
                    s.encode(encoding='utf-8').decode('ascii')
                except UnicodeDecodeError:
                    return False
                else:
                    return True
                
            


            if keywords is None:
                related_documents = self.indexer.similarity_search(question_with_chat_history, index_name=index_name)
            else:
                # check if keywords contains non english characters
                # if it does, then do not use the keywords
                if isEnglish(keywords):
                    related_documents = self.indexer.similarity_search(question_with_chat_history, index_name=index_name, search_text=keywords)
                else:
                    related_documents = self.indexer.similarity_search(question_with_chat_history, index_name=index_name)
            
            related_documents_filtered = []
            for document, score in related_documents:
                if score >= self.config.DOCUMENT_SIMILARITY_THRESHOLD:
                    related_documents_filtered.append(document)
            # concatenate the documents
            documents = self.concatenate_documents(related_documents_filtered)

            # TODO: add parameter to set llm light or not
            agent = get_doc_retrieval_agent(self.llm)

            executor = AgentExecutor(agent)

            answer = executor.run(question=question, chat_history=chat_history, summary=documents)['output']


        elif intent == "MDRT_QnA":
            agent = get_mdrt_qna_agent(self.llm)

            executor = AgentExecutor(agent)

            answer = executor.run(question=question, chat_history=chat_history)['output']


        else:
            # get a chat completion from the formatted messages
            logger.info("Getting answer from LLM without external knowledge")
            answer = self.llm(
                self.default_prompt.format_prompt(
                        chat_history=chat_history,
                        question=question, 
                    ).to_messages()
                ).content

        # Add the question and answer to the history

        question_message = message
        question_message.sequence_num = max_sequence_num + 1
        question_message.received_timestamp = str(received_timestamp)
        question_message.responded_timestamp = str(datetime.now())
        question_message.is_bot = 0

        answer_message = Message(
            text=answer,
            session_id=message.session_id,
            sequence_num=max_sequence_num + 2,
            received_timestamp=str(received_timestamp),
            responded_timestamp=str(datetime.now()),
            user_id=message.user_id,
            is_bot=1
        )

        # TODO: Dont contain source name in the chat history, otherwise it will 
        # confuse the model when generating the answer. Should log the original answer in long term store in the future

        # The source name is in the format of: [[file_name]]
        # e.g. [[A.txt]], [[https://test.blob.core.windows.net/test/A.txt]]

        # Remove the source name from the answer
        answer_removed_source_name = copy.deepcopy(answer_message)
        answer_removed_source_name.text = re.sub(r'\[\[.*?\]\]', '', answer_message.text)

        self.history_manager.add_qa_pair(question_message, answer_removed_source_name)

        source = Source()
        if intent == "DocRetrieval":
            logger.debug(f'Answer before replace citation: {answer_message.text}')
            answer_message, source = self.insert_citations_into_answer(answer_message)
            logger.debug(f'Answer after replace citation: {answer_message.text}')

        

        return Answer(answer_message, source)

    def _get_semantic_answer_langchain(
            self, 
            message: Message, 
            index_name: str = None, 
            condense_question: bool = True
        ) -> Answer:
        """
        Get the semantic answer using langchain.
        
        Args:
            question: the question
            index_name: the index name
            condense_question: whether to condense the question

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

        # Get the chat history
        chat_history = self.get_chat_history(question, message.session_id, self.config.CHAT_HISTORY_SEARCH_TYPE)

        chat_history = [c.text for c in chat_history] 

        def get_chat_history(inputs) -> str:
            """Get the chat history.
            
            Args:
                inputs: the inputs
            """

            chat_history_concatenated = '\n'.join(chat_history)
            return chat_history_concatenated

        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        
        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", verbose=True, prompt=PROMPT)
        chain = ConversationalRetrievalChain(
            retriever=self.indexer.get_retriever(index_name=index_name),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
            get_chat_history=get_chat_history,
            verbose=True,
            # top_k_docs_for_context= self.k
        )

        result = chain({"question": question, "chat_history": chat_history})

        # Add the question and answer to the history

        question_message = message
        question_message.sequence_num = max_sequence_num + 1
        question_message.received_timestamp = str(received_timestamp)
        question_message.responded_timestamp = str(datetime.now())
        question_message.is_bot = 0

        result['answer'] = result['answer'].split('SOURCES:')[0].split('Sources:')[0].split('SOURCE:')[0].split('Source:')[0]

        answer_message = Message(
            text=result['answer'],
            session_id=message.session_id,
            sequence_num=max_sequence_num + 2,
            received_timestamp=str(received_timestamp),
            responded_timestamp=str(datetime.now()),
            user_id=message.user_id,
            is_bot=1
        )

        self.history_manager.add_qa_pair(question_message, answer_message)

        return Answer(answer_message, None)
        

    def get_semantic_answer(
            self, 
            message: Message, 
            index_name: str = None, 
            condense_question: bool = True,
            conversation: str = 'custom' # 'custom' or 'langchain'
        ) -> Answer:
        """
        Get the semantic answer.

        Args:
            question: the question
            index_name: the index name
            condense_question: whether to condense the question
            conversation: the conversation type
        Returns:
            the answer
        """

        if conversation == 'custom':
            return self._get_semantic_answer_custom(message, index_name, condense_question)
        
        elif conversation == 'langchain':
            return self._get_semantic_answer_langchain(message, index_name, condense_question)
        
        else:
            raise ValueError('Conversation type not supported')
    
    def concatenate_documents(self, documents: List[Document]) -> str:
        """Concatenate the documents.
        
        Args:
            documents: the documents
        """

        result = ''

        for i in range(0, len(documents)):
            result += f"Content: {documents[i].page_content}\n"

            # Print the first 100 characters of the document
            logger.info(f"Content: {documents[i].page_content[:100]}")
            
            result += f"Source name: {documents[i].metadata['source']}\n\n"

            logger.info(f"Source name: {documents[i].metadata['source']}")

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


    def insert_citations_into_answer(self, answer: Message) -> Tuple[Message, Source]:
        """
        Replace the source names with citations in the answer.
        Return the source list as a Source object.

        Args:
            answer: the answer
        Returns:
            the new answer and the source list
        """

        # The source name is in the format of: [[file_name]]
        # e.g. [[A.txt]], [[https://test.blob.core.windows.net/test/A.txt]]

        # Get the source names
        source_names = re.findall(r'\[\[.*?\]\]', answer.text)

        # Get the source urls
        source_urls = [source_name[2:-2] for source_name in source_names]

        # remove duplications
        source_urls = list(set(source_urls))

        # Map the source urls to be an index starting from 1
        source_url_index = {source_url: i+1 for i, source_url in enumerate(source_urls)}

        # Replace the source name with citations
        for source_name in source_names:
            answer.text = answer.text.replace(source_name, f'[{source_url_index[source_name[2:-2]]}]')

        # Reverse the source url index
        source_url_index = {v: k for k, v in source_url_index.items()}

        return answer, Source(source_url_index)
    
    def get_all_chat_history(self, session_id: str) -> List[Message]:
        """Get the chat history for a session.
        
        Args:
            session_id: the session id
        """

        return self.history_manager.get_all_messages(session_id)
    
