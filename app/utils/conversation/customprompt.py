import logging

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_message_template = """You are a helpful assistant that can rephase a question based on the past conversation.
If the question is a follow up question, you need to rephrase the question to be a standalone question. 
The standalone question must contain all the information needed to answer it. Do not change the meaning of the question.
If the question is a standalone question already, you just repeat the question.
"""

human_question_template ="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

SYSTEM_MESSAGE_PROMPT = SystemMessagePromptTemplate.from_template(system_message_template)
HUMAN_MESSAGE_PROMPT = HumanMessagePromptTemplate.from_template(human_question_template)
