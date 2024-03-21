# # LLM
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# # Tools
# from langchain.agents import tool

# @tool
# def get_word_length(word: str) -> int:
#     """Returns the length of a word."""
#     return len(word)

# # Example tool invocation (not necessary for the agent to work, just for testing)
# get_word_length.invoke("abc")
# tools = [get_word_length]

# # Prompt
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# # Note: Adjusted to use a single prompt template for simplicity
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a very powerful assistant, but don't know current events."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# # LLM with Tools
# llm_with_tools = llm.bind_tools(tools)

# # Agent
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
#         "chat_history": lambda x: x.get("chat_history", []),
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIToolsAgentOutputParser()
# )

# from langchain.agents import AgentExecutor

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Chat History Management
# from langchain_core.messages import AIMessage, HumanMessage

# chat_history = []

# def invoke_agent_with_history(input_text):
#     global chat_history  # Use global or manage state more appropriately in your context
#     # Format chat_history for inclusion in the prompt
#     formatted_chat_history = [str(msg) for msg in chat_history]
#     # Invoke the agent with the current input and chat history
#     result = agent_executor.invoke({"input": input_text, "chat_history": formatted_chat_history})
#     # Update chat_history with the new interaction
#     chat_history.extend([
#         HumanMessage(content=input_text),
#         AIMessage(content=result["output"]),
#     ])
#     return result["output"]

# # Example usage
# input1 = "How many letters in the word educa?"
# response1 = invoke_agent_with_history(input1)
# print("Agent Response:", response1)

# input2 = "Is that a real word?"
# response2 = invoke_agent_with_history(input2)
# print("Agent Response:", response2)