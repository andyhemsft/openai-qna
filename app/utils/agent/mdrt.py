import re
import json
import logging
from typing import Any, Union, List, Dict, Tuple
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from app.utils.agent.base import BaseAgent, BaseTool, AgentAction, AgentFinish, AgentOutputParser
from app.utils.rdb import RelationDB
from app.config import Config


logger = logging.getLogger(__name__)

class GetMDRTReqTool(BaseTool):

    def __init__(self, 
                 llm, 
                 name: str = 'get_mdrt_req_tool', 
                 description: str = 'useful when you need to get the MDRT/COT/TOT requirement for a specific region'
                 ) -> None:
        
        self.chat_prompt = None
        self.args = {
            "category": "The category of the requirement, must be one of these values: MDRT, COT, TOT",
            "region": "The region of the agent, must be one of these values: HK, Macau"
        }
        
        self.db = RelationDB(Config())
        super().__init__(llm, name, description)

    def run(self, action_input: str, **kwargs: Any) -> str:
        
        # TODO: Get the category and region from the action_input
        # action_input = json.loads(action_input)
        category = action_input['category']
        region = action_input['region']
        # SQL query to get the MDRT requirement for a specific region
        query = f"SELECT Region, Category, FYCC, FYP, Case_Count FROM MDRT_Requirement WHERE Region = '{region}' AND Category = '{category}'"

        result = self.db.execute(query)

        answer = f"To achieve {category}, the agent must surpass the {region} thresholds requirement for {result[0][2]} FYCC, {result[0][3]} FYP, {result[0][4]} Case Count"
        
        return answer


class GetAgentSalesTool(BaseTool):

    def __init__(self, 
                 llm, 
                 name: str = 'get_agent_sales_tool', 
                 description: str = 'useful when you need to get the region and sales of an agent'
                 ) -> None:
        
        self.chat_prompt = None
        self.args = {
            "agent_name": "The name of the agent"
        }
        self.db = RelationDB(Config())
        super().__init__(llm, name, description)

    def run(self, action_input: str, **kwargs: Any) -> str:
        
        # TODO: Get the agent_name afrom the action_input
        agent_name = action_input['agent_name']

        # SQL query to get sales data of an agent
        query = f"SELECT Name, Region, FYCC, FYP, Case_Count FROM Agent_Sales WHERE Name = '{agent_name}'"

        result = self.db.execute(query)

        answer = f"{agent_name} is in {result[0][1]} Region, his sales record is: {result[0][2]} FYCC, {result[0][3]} FYP, {result[0][4]} Case Count"
        
        return answer

class CheckRequirementMeetTool(BaseTool):
    def __init__(self, 
                 llm, 
                 name: str = 'check_requirement_meet_tool', 
                 description: str = 'useful when you need to check if an agent meet the MDRT/COT/TOT requirement, you dont need to get the requirement and sales first, this tool will do it for you'
                 ) -> None:
        
        self.chat_prompt = None
        self.args = {
            "category": "The category of the requirement, must be one of these values: MDRT, COT, TOT",
            "agent_name": "The name of the agent"
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

        region = result[0][1]
        fycc = result[0][2]
        fyp = result[0][3]
        case_count = result[0][4]

        # SQL query to get the MDRT requirement for a specific region
        query = f"SELECT Region, Category, FYCC, FYP, Case_Count FROM MDRT_Requirement WHERE Region = '{region}' AND Category = '{category}'"

        result = self.db.execute(query)

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
                 llm, 
                 name: str = 'get_task_recommendation_tool', 
                 description: str = 'useful when you need to get the recommended tasks for an agent'
                 ) -> None:
        
        self.chat_prompt = None
        super().__init__(llm, name, description)


from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import BaseChatPromptTemplate

# Set up the base template
system_message_template = """Respond to the user as helpfully and accurately as possible. You have access to the following tools:

{tools}

To specify a tool, provide a JSON blob with an “action” key (tool name) and an “action_input” key (tool input).

Valid “action” values: “Final Answer” or {tool_names}

Provide only ONE action per JSON blob, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}

Begin! Remember to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. 
The format is Action:```$JSON_BLOB```then Observation:. The final response to the user may not always be the result of the last action, 
but should be based on the most helpful and accurate information obtained through the process."""

human_message_template = """
Chat History: {chat_history}
Question: {input}
{agent_scratchpad}"""

class AgentPromptTemplate:
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.system_message_template = system_message_template
        self.human_message_template = human_message_template

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""

        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        agent_scratchpad = thoughts
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    def format_prompt(self, **kwargs) -> str:

        prompt = self._create_prompt(**kwargs)

        return prompt.format_prompt(**kwargs)

    def _create_prompt(self, **kwargs) -> str:

        tool_strings = []
        for tool in self.tools:
            args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
            tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in self.tools])

        system_message_template = self.system_message_template.format(tools=formatted_tools, tool_names=tool_names)
        SYSTEM_MESSAGE_PROMPT = SystemMessagePromptTemplate.from_template(system_message_template)

        HUMAN_MESSAGE_PROMPT = HumanMessagePromptTemplate.from_template(human_message_template)

        return ChatPromptTemplate(messages=[SYSTEM_MESSAGE_PROMPT, HUMAN_MESSAGE_PROMPT], input_variables=["chat_history", "input", "agent_scratchpad"])

class MDRTQnAParser(AgentOutputParser):

    def __init__(self) -> None:
        
        self.pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            action_match = self.pattern.search(llm_output)
            if action_match is not None:
                response = json.loads(action_match.group(1).strip(), strict=False)
                if isinstance(response, list):
                    # gpt turbo frequently ignores the directive to emit a single action
                    logger.warning("Got multiple action responses: %s", response)
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, llm_output)
                else:
                    return AgentAction(
                        response["action"], response.get("action_input", {}), llm_output
                    )
            else:
                return AgentFinish({"output": llm_output}, llm_output)
        except Exception as e:
            raise ValueError(f"Could not parse LLM output: {llm_output}")


class MRDTQnAAgent(BaseAgent):

    def __init__(self, llm, tools) -> None:

        super().__init__(llm, tools)

        self.agent_output_parser = MDRTQnAParser()
        self.prompt = AgentPromptTemplate(
            tools=tools
        )

    def plan(self, 
            question: str, 
            chat_history: str, 
            intermediate_steps: List[Tuple[AgentAction, str]] = None,
            **kwargs: Any) -> Union[AgentFinish, AgentAction]:
        
        agent_scratchpad = self.prompt._construct_scratchpad(intermediate_steps)

        logger.info(f"{agent_scratchpad}")

        output = self.llm(self.prompt.format_prompt(
                input=question,
                chat_history=chat_history,
                agent_scratchpad=agent_scratchpad
            ).to_messages()
        ).content


        output = self.agent_output_parser.parse(output)

        if isinstance(output, AgentFinish):
            logger.info(f"Final Answer: {output.return_values['output']}")

        return output


def get_mdrt_qna_agent(llm):
    
    mdrt_req_tool = GetMDRTReqTool(llm)
    agent_sales_tool = GetAgentSalesTool(llm)
    check_requirement_meet_tool = CheckRequirementMeetTool(llm)

    mdrt_qna_agent = MRDTQnAAgent(llm=llm, tools=[mdrt_req_tool, agent_sales_tool, check_requirement_meet_tool])

    return mdrt_qna_agent