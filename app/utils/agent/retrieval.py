import re
import logging
from typing import Any, Union, List, Dict, Tuple
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from app.utils.agent.base import BaseAgent, BaseTool, AgentAction, AgentFinish, AgentOutputParser

logger = logging.getLogger(__name__)

# For get semantic answer based on the related documents with chat history
system_message_template_qa_w_history = """You are an adept assistant, capable of answering questions based on context user provided.
Please reply to the question using only the information presented in the summary and prior conversations.
Always include the source name for each fact you use in the response (e.g.: [[sample/A.txt]], [[doc/B.pdf]], etc.) to support your idea. Dont make up any source name.
If you can't find it, reply politely that the information is not in the knowledge base. 
Detect the language of the last question and answer in the same language. 
If asked for enumerations list all of them and do not invent any.
If the question is a greeting, reply with a greeting and ask what I can help.
"""
#Try to provide an answer as concise as possible, but do not sacrifice the quality of the answer.

human_question_template_qa_w_history = """
Summary: 
{summary}
Prior conversations: 
{chat_history}
Human:{question}
AI:"""

SYSTEM_MESSAGE_PROMPT_QA_W_HISTORY = SystemMessagePromptTemplate.from_template(system_message_template_qa_w_history)
HUMAN_MESSAGE_PROMPT_QA_W_HISTORY = HumanMessagePromptTemplate.from_template(human_question_template_qa_w_history)

chat_prompt = ChatPromptTemplate.from_messages(
    [SYSTEM_MESSAGE_PROMPT_QA_W_HISTORY, HUMAN_MESSAGE_PROMPT_QA_W_HISTORY]
)

class DocRetrievalParser(AgentOutputParser):

    def __init__(self) -> None:
        pass

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class DocRetrivalTool(BaseTool):

    def __init__(self, 
                 llm, 
                 name: str = 'doc_retrival_tool', 
                 description: str = 'useful when you need to answer a question based on the related documents'
                 ) -> None:
        
        self.chat_prompt = chat_prompt
        super().__init__(llm, name, description)

    def run(self, 
            question: str, 
            chat_history: str, 
            **kwargs: Any) -> str:
        if "summary" in kwargs:
            summary = kwargs["summary"]

        else:
            summary = ""

        # Each paragraph starts with keyword Conntent:
        # Find all the paragraphs
        # paragraphs = re.findall(r"Content:.*", summary)

        # # Each Source starts with keyword Source name:
        # # Find all the sources
        # sources = re.findall(r"Source name:.*", summary)

        # assert len(paragraphs) == len(sources), "The number of paragraphs and sources should be the same"
        
        # Take first 100 characters of each paragraph
        # snippets = "\n".join([f"{paragraph[:100]}......" for paragraph in paragraphs])

        # Take first 100 characters of each source 


        logger.info(f"Running agent with summary: {summary[:200]} ......")
        # get a chat completion from the formatted messages
        answer = self.llm(
            self.chat_prompt.format_prompt(
                    summary=summary,
                    chat_history=chat_history,
                    question=question, 
                ).to_messages()
            ).content
        
        return answer

class DocRetrivalAgent(BaseAgent):

    def __init__(self, llm, tools) -> None:
        self.prompt = None
        self.agent_output_parser = DocRetrievalParser()
        
        super().__init__(llm, tools)

    # def add_tool(self, tool: BaseTool) -> None:
    #     self.tools.append(tool)

    def plan(self, 
            question: str, 
            chat_history: str, 
            intermediate_steps: List[Tuple[AgentAction, str]] = None,
            **kwargs: Any) -> Union[AgentFinish, AgentAction]:
        
        if "summary" in kwargs:
            summary = kwargs["summary"]
        
        else:
            summary = ""

        # get a chat completion from the formatted messages
        output = self.tools[0].run(
            question=question, 
            chat_history=chat_history, 
            summary=summary
        )
        
        # Only need one run to get the answer
        output = f"Final Answer: {output}"

        # parse the output
        return self.agent_output_parser.parse(output)
    
def get_doc_retrieval_agent(llm):
    doc_retrival_tool = DocRetrivalTool(llm)
    doc_retrival_agent = DocRetrivalAgent(llm, [doc_retrival_tool])
    return doc_retrival_agent
