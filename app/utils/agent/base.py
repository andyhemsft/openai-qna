import os
import re
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class BaseTool:

    def __init__(self, 
                 llm, 
                 name: str, 
                 description: str,
                 ) -> None:
        
        self.llm = llm
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, 
            question: str, 
            chat_history: str, 
            **kwargs: Any
        ) -> str:
        """This function runs the tool."""

class BaseAgent:

    def __init__(self, llm, tools) -> None:
        self.llm = llm
        self.tools = tools

class AgentOutputParser:

    def __init__(self) -> None:
        pass

class AgentAction:

    def __init__(self, tool: str, tool_input: str, log: str) -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class AgentFinish:

    def __init__(self, return_values: Dict[str, str], log: str) -> None:
        self.return_values = return_values
        self.log = log

class AgentObsevation:

    def __init__(self) -> None:
        pass    
        
class AgentExecutor:

    def __init__(self, 
                 agent: BaseAgent, 
                 return_intermediate_steps: bool = False,
                 max_iterations: Optional[int] = 15,
                 max_execution_time: Optional[float] = None,
                 early_stopping_method: str = "force",
                 verbose: bool = False,
                 ) -> None:
        self.agent = agent
        self.tools = agent.tools
        self.return_intermediate_steps = return_intermediate_steps
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.early_stopping_method = early_stopping_method
        self.verbose = verbose
        self.name_to_tool_map = {tool.name: tool for tool in self.tools}

    def _take_next_action(self,
                        question : str,
                        chat_history : str,
                        intermediate_steps: List[Tuple[AgentAction, str]],
                        **kwargs: Any
        ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:

        intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                question=question,
                chat_history=chat_history,
                intermediate_steps=intermediate_steps,
                **kwargs,
            )

            
        except Exception as e:
            raise ValueError(f"Error in agent plan: {e}")
        
        # Parse the output
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        
        # If the tool chosen is an action, then we run the tool and update the inputs.
        if isinstance(output, AgentAction):
            actions = [output]

        result = []
        for agent_action in actions:

            if agent_action.tool not in self.name_to_tool_map:
                raise ValueError(f"Tool `{agent_action.tool}` is not in the list of tools.")
            
            else:
                # Get the tool
                tool = self.name_to_tool_map[agent_action.tool]

                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                )
            
            result.append((agent_action, observation))
        
        return result

    def _prepare_intermediate_steps(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> List[Tuple[AgentAction, str]]:
        return intermediate_steps

    def run(self, question, chat_history, **kwargs) -> str:
        
        logger.info(f"Running agent with chat history: {chat_history}")
        logger.info(f"Running agent with question: {question}")
        
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        for i in range(self.max_iterations):
            next_step_output = self._take_next_action(question=question, 
                                                      chat_history=chat_history,
                                                      intermediate_steps=intermediate_steps,
                                                      **kwargs)

            if isinstance(next_step_output, AgentFinish):
                return next_step_output.return_values

            intermediate_steps.extend(next_step_output)

