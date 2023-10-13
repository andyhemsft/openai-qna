import logging

from app.config import Config
from app.utils.llm import LLMHelper
from langchain.prompts import PromptTemplate

from typing import Any, Union, List, Dict, Tuple
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



logger = logging.getLogger(__name__)


system_message_template = """
Here is the list of {intent}.
1. DocRetrieval
2. MDRT_QnA
3. Other

Extract the intent from the text started and ended by '''. The output will be in the intent name. Dont translate the language.
Here are some example.

Input:
'''I want to know the benific of this product'''

Output:
DocRetrieval

Input:
'''它提供什麼服務？'''

Output:
DocRetrieval

Input:
'''What is HSBC Life Insurance'''

Output:
DocRetrieval

Input:
'''What is fee for this product'''

Output:
DocRetrieval

Input:
'''What is the sales of Agent A'''

Output:
MDRT_QnA


Input:
'''What is the MDRT requirement'''

Output:
MDRT_QnA


Input:
'''Does Agent A meet the MDRT requirement'''

Output:
MDRT_QnA

Input:
'''Hello'''

Output:
Other

Input:
'''Tell me about AIA'''

Output:
Other
"""



human_message_template = """
Input:
'''{question}'''

Output:"""


SYSTEM_MESSAGE_PROMPT = SystemMessagePromptTemplate.from_template(system_message_template.format(intent="intent"))
HUMAN_MESSAGE_PROMPT = HumanMessagePromptTemplate.from_template(human_message_template)

# chat_prompt = ChatPromptTemplate.from_messages([SYSTEM_MESSAGE_PROMPT, HUMAN_MESSAGE_PROMPT])
chat_prompt = ChatPromptTemplate(messages=[SYSTEM_MESSAGE_PROMPT, HUMAN_MESSAGE_PROMPT], input_variables=["question"])

intent_detection_template = """You are an adept assistant, capable of discerning the purpose of a query and selecting the appropriate tool for a response. Here are the tools at your disposal:

DocRetrieval: This tool is beneficial when you need to respond to inquiries about HSBC products or services based on associated documents.
MDRT_QnA: This tool is advantageous when you need to respond to inquiries about MDRT and agent sales.
Other: This tool is helpful when you need to respond to inquiries that do not fall into the above two categories.
Question: {question} The most suitable tool for answering the question is:"""

# After answering the question generate three very brief follow-up questions that the user would likely ask next.
# Only use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
# Only generate questions and do not generate any text before or after the questions, such as 'Follow-up Questions:'.
# Try not to repeat questions that have already been asked.

INTENT_DETECTION_PROMPT = PromptTemplate(template=intent_detection_template, input_variables=["question"])

class IntentDetector:

    def __init__(self, config: Config) -> None:
        self.config = config


class LLMIntentDetector(IntentDetector):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.llm = LLMHelper(self.config).get_llm()
        self.chat_prompt = INTENT_DETECTION_PROMPT

    def detect_intent(self, question: str) -> str:

        answer = self.llm(self.chat_prompt.format_prompt(question=question).to_messages()).content

        return answer