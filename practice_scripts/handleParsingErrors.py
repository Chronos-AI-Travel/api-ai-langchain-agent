from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import OpenAI

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [tool]

prompt = hub.pull("hwchase17/react")

llm = OpenAI(temperature=0)

agent = create_react_agent(llm, tools, prompt)

def _handle_error(error) -> str:
    return str(error)[:50]


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=_handle_error,
)

agent_executor.invoke(
    {"input": "What is Leo DiCaprio's middle name?\n\nAction: Wikipedia"}
)

