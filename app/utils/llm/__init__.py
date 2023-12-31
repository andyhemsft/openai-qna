import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from app.config import Config

class LLMHelper:
    def __init__(self, config: Config):
        '''
            Initialize the LLM helper
            Args:
                config: the config object
        '''

        self.config = config

    def get_llm(self, temperature: float = None) -> ChatOpenAI:
        '''
            Returns the LLM model based on the config

            Args:
                none
            Returns:
                the LLM model
        '''

        if self.config.LLM_TYPE == 'openai':
            assert self.config.OPENAI_API_TYPE in ['azure', 'openai'], 'OPENAI_API_TYPE must be either azure or openai'
            assert self.config.OPENAI_API_BASE is not None, 'OPENAI_API_BASE must be set'
            assert self.config.OPENAI_API_KEY is not None, 'OPENAI_API_KEY must be set'
            assert self.config.OPENAI_ENGINE is not None, 'OPENAI_ENGINE must be set'

            if temperature is None:
                temperature = self.config.OPENAI_TEMPERATURE

            # We should use the chat completion API, since azure GPT-4 only supports chat completion
            return ChatOpenAI(
                model_name = self.config.OPENAI_ENGINE,
                engine = self.config.OPENAI_ENGINE,
                temperature = temperature,
                max_tokens = self.config.OPENAI_MAX_TOKENS,
            )

        else:
            raise ValueError('LLM type not supported')

    def get_embeddings(self) -> OpenAIEmbeddings:
        '''
            Returns the LLM embedding based on the config.

            Args:
                none
            Returns:
                the LLM embedding
        '''

        if self.config.LLM_TYPE == 'openai':
            assert self.config.OPENAI_API_TYPE in ['azure', 'openai'], 'OPENAI_API_TYPE must be either azure or openai'
            assert self.config.OPENAI_API_BASE is not None, 'OPENAI_API_BASE must be set'
            assert self.config.OPENAI_API_KEY is not None, 'OPENAI_API_KEY must be set'
            assert self.config.OPENAI_EMBEDDING_ENGINE is not None, 'OPENAI_EMBEDDING_ENGINE must be set'

            return OpenAIEmbeddings(
                model_name = self.config.OPENAI_EMBEDDING_ENGINE,
                deployment = self.config.OPENAI_EMBEDDING_ENGINE,
                chunk_size = 1,
                disallowed_special = () # Allow all special tokens
            )

        else:
            raise ValueError('LLM type not supported')
