import logging

from app.config import Config
from app.utils.llm import LLMHelper

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

class IntentDetector:

    def __init__(self, config: Config) -> None:
        self.config = config


class LLMIntentDetector(IntentDetector):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.llm = LLMHelper(self.config).get_llm()
        self.chat_prompt = chat_prompt

    def detect_intent(self, question: str) -> str:

        answer = self.llm(self.chat_prompt.format_prompt(question=question).to_messages()).content

        return answer