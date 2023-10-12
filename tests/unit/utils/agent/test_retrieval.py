import logging
import pytest

from app.utils.agent.base import AgentFinish, AgentExecutor
from app.utils.agent.retrieval import DocRetrivalTool, DocRetrivalAgent
from app.utils.llm import LLMHelper
from app.config import Config

logger = logging.getLogger(__name__)


@pytest.fixture
def doc_retrival_tool():

    config = Config()
    llm = LLMHelper(config).get_llm()

    return DocRetrivalTool(llm)

@pytest.fixture
def doc_retrival_agent(doc_retrival_tool):

    doc_retrival_agent = DocRetrivalAgent(llm=None, tools=[doc_retrival_tool])
    
    return doc_retrival_agent


@pytest.fixture
def agent_executor(doc_retrival_agent):
    
    return AgentExecutor(doc_retrival_agent)

def test_doc_retrival_tool(doc_retrival_tool):
    
    question = "What is the capital of the United States?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"
    summary = "Content: The capital of the United States is Washington, D.C."

    answer = doc_retrival_tool.run(question, chat_history, summary=summary)

    assert "Washington, D.C." in answer

def test_doc_retrival_agent(doc_retrival_agent):

    question = "What is the capital of the United States?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"
    summary = "Content: The capital of the United States is Washington, D.C."

    answer = doc_retrival_agent.plan(question, chat_history, summary=summary)

    assert isinstance(answer, AgentFinish)
    assert "Washington, D.C." in answer.return_values['output']


def test_agent_executor(agent_executor):

    # Test case 1
    question = "What is the capital of the United States?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"
    summary = "Content: The capital of the United States is Washington, D.C."

    output = agent_executor.run(question=question, chat_history=chat_history, summary=summary)

    assert "Washington, D.C." in output['output']



