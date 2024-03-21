# from langchain import hub
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_openai import ChatOpenAI

# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
# tool = WikipediaQueryRun(api_wrapper=api_wrapper)
# tools = [tool]

# prompt = hub.pull("hwchase17/openai-functions-agent")

# llm = ChatOpenAI(temperature=0)

# agent = create_openai_functions_agent(llm, tools, prompt)

# agent_executor = AgentExecutor(
#     agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
# )

# response = agent_executor.invoke({"input": "What is Leo DiCaprio's middle name?"})

# # The actual return type is a NamedTuple for the agent action, and then an observation
# print(response["intermediate_steps"])