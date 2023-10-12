import logging

from langchain.prompts import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# For rephase question
system_message_template_rephrase_q = """You are a proficient assistant, skilled in rephrasing questions based on prior conversations. 
If a follow-up question is already self-contained, you simply repeat it. 
If the rephrased question conveys the same meaning as the original question, you just repeat the original question. 
However, if the follow-up question lacks context, you expertly rephrase it into a standalone question that includes all the necessary information for a complete answer.
"""

human_question_template_rephrase_q = """
Prior conversations:
{chat_history}
Follow up question: {question}
Rephased question:"""

SYSTEM_MESSAGE_PROMPT_REPHRASE_Q = SystemMessagePromptTemplate.from_template(system_message_template_rephrase_q)
HUMAN_MESSAGE_PROMPT_REPHRASE_Q = HumanMessagePromptTemplate.from_template(human_question_template_rephrase_q)

# For rephase question
system_message_template_rephrase_keyword = """You are a proficient assistant, skilled in rephrasing questions and extracting the keywords based on prior conversations. 
If a follow-up question is already self-contained, you simply repeat it. 
If the rephrased question conveys the same meaning as the original question, you just repeat the original question. 
However, if the follow-up question lacks context, you expertly rephrase it into a standalone question that includes all the necessary information for a complete answer.
Always provide the keywords for the question in the output. The rephased question must be in the same language as the original question.
The keywords must be translate into English.

The output will be in the following format:
This is the rephrased question in original language. [[[keyword1 in english, keyword2 in english, keyword3 in english]]]
"""



human_question_template_rephrase_keyword = """
Prior conversations:
{chat_history}
Follow up question: {question}
Output:"""

SYSTEM_MESSAGE_PROMPT_REPHRASE_KEYWORD = SystemMessagePromptTemplate.from_template(system_message_template_rephrase_keyword)
HUMAN_MESSAGE_PROMPT_REPHRASE_KEYWORD = HumanMessagePromptTemplate.from_template(human_question_template_rephrase_keyword)

# For get semantic answer based on the related documents
system_message_template_qa_wo_history = """You are an adept assistant, capable of answering questions based on context user provided.
Please reply to the question using only the information presented in the summary.
Always include the source name for each fact you use in the response (e.g.: [[sample/A.txt]], [[doc/B.pdf]], etc.) to support your idea.
If you can't find it, reply politely that the information is not in the knowledge base.
Detect the language of the question and answer in the same language. 
If asked for enumerations list all of them and do not invent any.
If the question is a greeting, reply with a greeting and ask what I can help.
"""

human_question_template_qa_wo_history = """
Summary: 
{summary}
Human:{question}
AI:"""

SYSTEM_MESSAGE_PROMPT_QA_WO_HISTORY = SystemMessagePromptTemplate.from_template(system_message_template_qa_wo_history)
HUMAN_MESSAGE_PROMPT_QA_WO_HISTORY = HumanMessagePromptTemplate.from_template(human_question_template_qa_wo_history)


# For get semantic answer based on the related documents with chat history
system_message_template_qa_w_history = """You are an adept assistant, capable of answering questions based on context user provided.
Please reply to the question using only the information presented in the summary and prior conversations.
Always include the source name for each fact you use in the response (e.g.: [[sample/A.txt]], [[doc/B.pdf]], etc.) to support your idea.
If you can't find it, reply politely that the information is not in the knowledge base.
Detect the language of the question and answer in the same language. 
If asked for enumerations list all of them and do not invent any.
If the question is a greeting, reply with a greeting and ask what I can help.
"""

human_question_template_qa_w_history = """
Summary: 
{summary}
Prior conversations: 
{chat_history}
Human:{question}
AI:"""

SYSTEM_MESSAGE_PROMPT_QA_W_HISTORY = SystemMessagePromptTemplate.from_template(system_message_template_qa_w_history)
HUMAN_MESSAGE_PROMPT_QA_W_HISTORY = HumanMessagePromptTemplate.from_template(human_question_template_qa_w_history)


template = """{summaries}

Please reply to the question using only the information present in the text above.
If you can't find it, reply politely that the information is not in the knowledge base.
Detect the language of the question and answer in the same language. 
If asked for enumerations list all of them and do not invent any.
Each source has a name followed by a colon and the actual information, always include the source name for each fact you use in the response. Always use double square brackets to reference the filename source, e.g. [[info1.pdf.txt]]. Don't combine sources, list each source separately, e.g. [[info1.pdf]][[info2.txt]].
If the question is a greeting, reply with a greeting and ask what I can help.

Question: {question}
Answer:"""

# After answering the question generate three very brief follow-up questions that the user would likely ask next.
# Only use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
# Only generate questions and do not generate any text before or after the questions, such as 'Follow-up Questions:'.
# Try not to repeat questions that have already been asked.

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])


default_template = """
Try your best to answer user's question. If you don't know the answer, just reply you don't know.
If the question is a greeting, reply with a greeting and ask what I can help.

Prior conversations: 
{chat_history}
Human:{question}
Answer:"""

# After answering the question generate three very brief follow-up questions that the user would likely ask next.
# Only use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
# Only generate questions and do not generate any text before or after the questions, such as 'Follow-up Questions:'.
# Try not to repeat questions that have already been asked.

DEFAULT_PROMPT = PromptTemplate(template=default_template, input_variables=["chat_history", "question"])


chat_summarization_template = """
please provide a detailed summary of the following interaction between the user and the AI assistant. 
Ensure to emphasize the main topic of their discussion for potential future reference.

Chat: 
{chat}
Summary:"""

# After answering the question generate three very brief follow-up questions that the user would likely ask next.
# Only use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
# Only generate questions and do not generate any text before or after the questions, such as 'Follow-up Questions:'.
# Try not to repeat questions that have already been asked.

CHAT_SUMMARIZATION_PROMPT = PromptTemplate(template=chat_summarization_template, input_variables=["chat"])