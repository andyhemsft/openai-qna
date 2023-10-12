import logging
import pytest

from app.config import Config
from app.utils.llm import LLMHelper
from app.utils.agent.base import AgentAction, AgentFinish, AgentExecutor
from app.utils.agent.mdrt import GetMDRTReqTool, GetAgentSalesTool, CheckRequirementMeetTool, MDRTQnAParser, MRDTQnAAgent, AgentPromptTemplate


logger = logging.getLogger(__name__)


@pytest.fixture
def get_mdrt_req_tool():

    config = Config()
    llm = LLMHelper(config).get_llm()
    return GetMDRTReqTool(llm=llm)

@pytest.fixture
def get_agent_sales_tool():

    config = Config()
    llm = LLMHelper(config).get_llm()
    return GetAgentSalesTool(llm=llm)

@pytest.fixture
def check_requirement_meet_tool():

    config = Config()
    llm = LLMHelper(config).get_llm()
    return CheckRequirementMeetTool(llm=llm)


@pytest.fixture
def agent_prompt_template(get_mdrt_req_tool):

    return AgentPromptTemplate([get_mdrt_req_tool])

@pytest.fixture
def mdrt_qna_parser():
    
    return MDRTQnAParser()

@pytest.fixture
def mdrt_qna_agent(get_mdrt_req_tool, get_agent_sales_tool, check_requirement_meet_tool):
    
    config = Config()
    llm = LLMHelper(config).get_llm()
    return MRDTQnAAgent(llm=llm, tools=[get_mdrt_req_tool, get_agent_sales_tool, check_requirement_meet_tool])

@pytest.fixture
def agent_executor(mdrt_qna_agent):

    return AgentExecutor(mdrt_qna_agent)

def test_get_mdrt_req_tool(get_mdrt_req_tool):
    pass

def test_get_agent_sales_tool(get_agent_sales_tool):

    action_input = {"agent_name": "Agent A"}

    result = get_agent_sales_tool.run(action_input)

    assert "Agent A" in result

def test_check_requirement_meet_tool(check_requirement_meet_tool):

    action_input = {"agent_name": "Agent A", "category": "MDRT"}

    result = check_requirement_meet_tool.run(action_input)

    logger.info(result)

    assert "Agent A" in result

# def test_mdrt_qna_parser(mdrt_qna_parser):

#     llm_output = """Action:
#     ```
#     {
#       "action": "get_mdrt_req_tool",
#       "action_input": {
#         "category": "MDRT",
#         "region": "HK"
#       }
#     }
#     ```
#     """

#     result = mdrt_qna_parser.parse(llm_output)

#     assert isinstance(result, AgentAction)
#     assert result.tool == "get_mdrt_req_tool"
#     assert result.tool_input == {'category': 'MDRT', 'region': 'HK'}

# def test_construct_scratchpad(mdrt_qna_parser, agent_prompt_template):

#     llm_output = """Action:
#     ```
#     {
#       "action": "get_mdrt_req_tool",
#       "action_input": {
#         "category": "MDRT",
#         "region": "HK"
#       }
#     }
#     ```
#     """

#     action = mdrt_qna_parser.parse(llm_output)

#     assert isinstance(action, AgentAction)
#     observation = "MDRT requirement for HK is 100 FYCC, 400 FYP, 5 Case Count"

#     intermediate_steps = [(action, observation)]

#     scratchpad = agent_prompt_template._construct_scratchpad(
#         intermediate_steps=intermediate_steps,
#     )

#     logger.info(scratchpad)

# def test_format_prompt(mdrt_qna_parser, agent_prompt_template):
#     """Test the AgentPromptTemplate class."""

#     llm_output = """Action:
#     ```
#     {
#       "action": "get_mdrt_req_tool",
#       "action_input": {
#         "category": "MDRT",
#         "region": "HK"
#       }
#     }
#     ```
#     """

#     action = mdrt_qna_parser.parse(llm_output)

#     assert isinstance(action, AgentAction)
#     observation = "MDRT requirement for HK is 100 FYCC, 400 FYP, 5 Case Count"

#     intermediate_steps = [(action, observation)]

#     agent_scratchpad = agent_prompt_template._construct_scratchpad(intermediate_steps)

#     formated_prompt = agent_prompt_template.format_prompt(
#         chat_history="Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n",
#         input="What is the MDRT requirement for Hong Kong?",
#         agent_scratchpad=agent_scratchpad
#     )

#     logger.info(formated_prompt.messages[0])

#     logger.info(formated_prompt.messages[1])

def test_mdrt_agent_plan(mdrt_qna_agent):

    question = "What is the MDRT requirement for Hong Kong?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"
    intermediate_steps = []

    output = mdrt_qna_agent.plan(question=question, chat_history=chat_history, intermediate_steps=intermediate_steps)
    
    assert isinstance(output, AgentAction)

    logger.info(f"Action: {output.tool}")
    logger.info(f"Action Input: {output.tool_input}")


def test_agent_executor(agent_executor):

    # Test case 1
    question = "What is the MDRT requirement for Hong Kong?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"

    output = agent_executor.run(question=question, chat_history=chat_history)

    logger.info(output)

    assert "requirement" in output['output']

    # Test case 2
    question = "What is the sales record for Agent A?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"

    output = agent_executor.run(question=question, chat_history=chat_history)

    logger.info(output)

    assert "Agent A" in output['output']

    # Test case 3
    question = "Does Agent A meet the MDRT requirement?"
    chat_history = "Human:Hi\nAI:Hello\nHuman:How are you?\nAI:I am good. How are you?\n"

    output = agent_executor.run(question=question, chat_history=chat_history)

    logger.info(output)



