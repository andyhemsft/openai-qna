import os
import openai

import re
import json
import logging
from typing import Any, Union, List, Dict, Tuple

from app.utils.agent.base import BaseAgent, BaseTool, AgentAction, AgentFinish, AgentOutputParser
from app.utils.rdb import RelationDB
from app.config import Config

import openai


logger = logging.getLogger(__name__)

class GetMDRTReqTool(BaseTool):

    def __init__(self, 
                 llm=None, 
                 name: str = 'get_mdrt_req_tool', 
                 description: str = 'useful when you need to get the MDRT/COT/TOT requirement for a specific region'
                 ) -> None:

        self.name = "get_mdrt_req_tool",
        self.description = "useful when you need to get the MDRT/COT/TOT requirement for a specific region",
        self.parameters = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category of the requirement, must be one of these values: MDRT, COT, TOT"
                },
                "region": {
                    "type": "string",
                    "description": "The region of the agent, must be one of these values: HK, Macau"
                }
            },
            "required": ["category", "region"]
        }

        
        self.db = RelationDB(Config())
        super().__init__(llm, name, description)

    def run(self, action_input: str, **kwargs: Any) -> str:
        
        # TODO: Get the category and region from the action_input
        # action_input = json.loads(action_input)
        category = action_input['category'].strip()
        region = action_input['region'].strip()
        # SQL query to get the MDRT requirement for a specific region
        query = f"SELECT Region, Category, FYCC, FYP, Case_Count FROM MDRT_Requirement WHERE Region = '{region}' AND Category = '{category}'"

        result = self.db.execute(query)

        if len(result) == 0:
            answer = f"Cannot find the requirement for {category} in {region}. Please provide the correct information."
            return answer

        logger.debug(f"SQL -- Running agent with category: {category}, region: {region}")
        logger.debug(f"SQL -- Result: {result}")
        answer = f"To achieve {category.strip()}, the agent must surpass the {region.strip()} thresholds requirement for {result[0][2]} FYCC, {result[0][3]} FYP, {result[0][4]} Case Count"
        
        return answer


class GetAgentSalesTool(BaseTool):

    def __init__(self, 
                 llm=None, 
                 name: str = 'get_agent_sales_tool', 
                 description: str = 'useful when you need to get the region and sales of an agent'
                 ) -> None:

        self.name = "get_agent_sales_tool",
        self.description = "useful when you need to get the region and sales of an agent"
        self.parameters = {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "The name of the agent"
                }
            },
            "required": ["agent_name"]
        }

        self.db = RelationDB(Config())
        super().__init__(llm, name, description)

    def run(self, action_input: str, **kwargs: Any) -> str:
        
        # TODO: Get the agent_name afrom the action_input
        agent_name = action_input['agent_name']

        # SQL query to get sales data of an agent
        query = f"SELECT Name, Region, FYCC, FYP, Case_Count FROM Agent_Sales WHERE Name = '{agent_name}'"

        result = self.db.execute(query)

        if len(result) == 0:
            answer = f"Cannot find your name: {agent_name} in the database. Please provide the correct information."
            return answer

        logger.debug(f"SQL -- Running agent with agent_name: {agent_name}")
        logger.debug(f"SQL -- Result: {result}")
        answer = f"{agent_name} is in {result[0][1]} Region, his sales record is: {result[0][2]} FYCC, {result[0][3]} FYP, {result[0][4]} Case Count"
        
        return answer

class CheckRequirementMeetTool(BaseTool):
    def __init__(self, 
                 llm=None, 
                 name: str = 'check_requirement_meet_tool', 
                 description: str = 'useful when you need to check if an agent meet the MDRT/COT/TOT requirement, you dont need to get the requirement and sales first, this tool will do it for you'
                 ) -> None:

        self.name = "check_requirement_meet_tool",
        self.description = "useful when you need to check if an agent meet the MDRT/COT/TOT requirement, you dont need to get the requirement and sales first, this tool will do it for you"
        self.parameters = {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "The name of the agent"
                },
                "category": {
                    "type": "string",
                    "description": "The category of the requirement, must be one of these values: MDRT, COT, TOT"
                }
            },
            "required": ["agent_name", "category"]
        }

        self.db = RelationDB(Config())
        super().__init__(llm, name, description)

    def run(self, action_input: str, **kwargs: Any) -> str:
        
        # TODO: Get the agent_name afrom the action_input
        category = action_input['category']
        agent_name = action_input['agent_name']

        # SQL query to get sales data of an agent
        query = f"SELECT Name, Region, FYCC, FYP, Case_Count FROM Agent_Sales WHERE Name = '{agent_name}'"

        result = self.db.execute(query)

        if len(result) == 0:
            answer = f"Cannot find your name: {agent_name} in the database. Please provide the correct information."
            return answer

        region = result[0][1]
        fycc = result[0][2]
        fyp = result[0][3]
        case_count = result[0][4]

        # SQL query to get the MDRT requirement for a specific region
        query = f"SELECT Region, Category, FYCC, FYP, Case_Count FROM MDRT_Requirement WHERE Region = '{region}' AND Category = '{category}'"

        result = self.db.execute(query)

        if len(result) == 0:
            answer = f"Cannot find the requirement for {category} in {region}. Please provide the correct information."
            return answer

        fycc_requirement = result[0][2]
        fyp_requirement = result[0][3]
        case_count_requirement = result[0][4]

        if fycc >= fycc_requirement and fyp >= fyp_requirement and case_count >= case_count_requirement:
            answer = f"""{agent_name} meets the {category} requirement. 
                        The {category} requirement in {region} is {fycc_requirement} FYCC, 
                        {fyp_requirement} FYP, {case_count_requirement} Case Count.
                        The agent's sales record is {fycc} FYCC, {fyp} FYP, {case_count} Case Count.
                        """

        else:
            answer = f"""{agent_name} does not meet the {category} requirement.
                        The {category} requirement in {region} is {fycc_requirement} FYCC, 
                        {fyp_requirement} FYP, {case_count_requirement} Case Count.
                        The agent's sales record is {fycc} FYCC, {fyp} FYP, {case_count} Case Count.
                        """
        
        return answer    


class GetTaskRecommendationTool(BaseTool):

    def __init__(self, 
                 llm=None, 
                 name: str = 'get_task_recommendation_tool', 
                 description: str = 'useful when you need to get the recommended tasks for an agent'
                 ) -> None:
        
        super().__init__(llm, name, description)


human_message_template = """
Chat History: {chat_history}
Question: {input}"""



openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")


class MRDTQnAAgent(BaseAgent):

    def __init__(self, llm=None, tools=None) -> None:
        

        self.functions_list = []
        self.functions_dict = {}

        for tool in tools:
            self.functions_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            )

            self.functions_dict[tool.name] = tool.parameters["required"]

        self.name_to_tool_map = {tool.name: tool for tool in tools}
        super().__init__(llm, tools)


    def plan(self, 
            question: str, 
            chat_history: str, 
            intermediate_steps: List[Tuple[AgentAction, str]] = None,
            **kwargs: Any) -> Union[AgentFinish, AgentAction]:
    
        
        messages = [
            {"role": "user", "content": human_message_template.format(chat_history=chat_history, input=question)}
        ]

        response = openai.ChatCompletion.create(
                            engine=os.getenv("OPENAI_ENGINE"),
                            messages=messages,
                            functions=self.functions_list,
                            function_call="auto", 
                        )
        
        # log the response
        logger.debug(f"Running agent with response: {response}")

        # only get the first message 
        response_message = response['choices'][0]['message']

        # if the response message is not a function call, return the response message
        if "function_call" not in response_message:
            answer = response_message['content']

            logger.info(f"Running agent with answer without function call: {answer}")

            return AgentFinish({"output": answer}, response)

        else:
            function_call_name = response['choices'][0]['message']['function_call']['name']

            logger.info(f"Running agent with function_call_name: {function_call_name}")

            function_call_arguments = response['choices'][0]['message']['function_call']['arguments']

            logger.debug(f"Running agent with function_call_arguments: {function_call_arguments}")

            # convert the function_call_arguments to a dictionary
            function_call_arguments = json.loads(function_call_arguments.replace("'", '"'))

            # Check if all required parameters are provided
            missing_arguments = []
            for required_arg in self.functions_dict[function_call_name]:
                if required_arg not in function_call_arguments:
                    missing_arguments.append(required_arg)

            if len(missing_arguments) > 0:
                # Ask user to provide the missing arguments
                answer = f"Please provide the missing arguments: {missing_arguments}"
                return AgentFinish({"output": answer}, response)

            else:
                # Get the tool
                tool = self.name_to_tool_map[function_call_name]

                answer = tool.run(
                    function_call_arguments
                )
            
                # Return AgentAction with the function_call_name and function_call_arguments
                return AgentFinish({"output": answer}, response)
        

def get_mdrt_qna_agent(llm=None):
    # initialize the tools
    get_mdrt_req_tool = GetMDRTReqTool()
    get_agent_sales_tool = GetAgentSalesTool()
    check_requirement_meet_tool = CheckRequirementMeetTool()

    mdrt_qna_agent = MRDTQnAAgent(tools=[get_mdrt_req_tool, get_agent_sales_tool, check_requirement_meet_tool])

    return mdrt_qna_agent