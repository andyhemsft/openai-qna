import logging
import pytest

from app.config import Config
from app.utils.llm import LLMHelper
from app.utils.agent.base import AgentAction, AgentFinish, AgentExecutor
from app.utils.agent.mdrt_function_call import GetMDRTReqTool, GetAgentSalesTool, CheckRequirementMeetTool, MRDTQnAAgent

@pytest.fixture
def get_mdrt_req_tool():

    return GetMDRTReqTool()

@pytest.fixture
def get_agent_sales_tool():

    return GetAgentSalesTool()

@pytest.fixture
def check_requirement_meet_tool():

    return CheckRequirementMeetTool()

@pytest.fixture
def mdrt_qna_agent(get_mdrt_req_tool, get_agent_sales_tool, check_requirement_meet_tool):
    
    return MRDTQnAAgent(tools=[get_mdrt_req_tool, get_agent_sales_tool, check_requirement_meet_tool])

@pytest.fixture
def agent_executor(mdrt_qna_agent):

    return AgentExecutor(mdrt_qna_agent)


def test_mdrt_qna_agent(mdrt_qna_agent):

    # Test get MDRT requirement
    chat_history = ""
    question = "What is the MDRT requirement?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    assert "the agent must surpass the" not in agent_output.return_values['output']

    chat_history = ""
    question = "What is the MDRT requirement in Hong Kong?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    # agent output should be AgentFinish
    assert isinstance(agent_output, AgentFinish)
    assert "the agent must surpass the" in agent_output.return_values['output']

    # Test get agent sales
    chat_history = ""
    question = "What is the agent sales?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    assert "sales record is" not in agent_output.return_values['output']

    chat_history = ""
    question = "What is the agent A sales in Hong Kong?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    # agent output should be AgentFinish
    assert isinstance(agent_output, AgentFinish)
    assert "sales record is" in agent_output.return_values['output']

    # Test check requirement meet
    chat_history = ""
    question = "Does the agent meet the MDRT requirement?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    assert "meets the MDRT requirement" not in agent_output.return_values['output']

    chat_history = ""
    question = "Does the agent A meet the MDRT requirement in Hong Kong?"

    agent_output = mdrt_qna_agent.plan(question, chat_history)

    # agent output should be AgentFinish
    assert isinstance(agent_output, AgentFinish)
    assert "the MDRT requirement" in agent_output.return_values['output']

def test_agent_executor(agent_executor):

    chat_history = ""
    question = "What is the MDRT requirement?"

    agent_output = agent_executor.run(question, chat_history)

    assert "the agent must surpass the" not in agent_output['output']

    chat_history = ""
    question = "What is the MDRT requirement in Hong Kong?"

    agent_output = agent_executor.run(question, chat_history)
    assert "the agent must surpass the" in agent_output['output']


