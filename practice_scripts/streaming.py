# from langchain import hub
# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain.prompts import ChatPromptTemplate
# from langchain.tools import tool
# from langchain_core.callbacks import Callbacks
# from langchain_openai import ChatOpenAI

# # LLM
# model = ChatOpenAI(temperature=0, streaming=True)

# # Tools
# import random


# @tool
# async def where_cat_is_hiding() -> str:
#     """Where is the cat hiding right now?"""
#     return random.choice(["under the bed", "on the shelf"])
#     # await where_cat_is_hiding.ainvoke({})


# # @tool
# # async def get_items(place: str) -> str:
# #     """Use this tool to look up which items are in the given place."""
# #     if "bed" in place:  # For under the bed
# #         return "socks, shoes and dust bunnies"
# #     if "shelf" in place:  # For 'shelf'
# #         return "books, penciles and pictures"
# #     else:  # if the agent decides to ask about a different place
# #         return "cat snacks"


# # Agent
# prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt.messages)
# tools = [get_items, where_cat_is_hiding]
# agent = create_openai_tools_agent(
#     model.with_config({"tags": ["agent_llm"]}), tools, prompt
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
#     {"run_name": "Agent"}
# )

# # Note: We use `pprint` to print only to depth 1, it makes it easier to see the output from a high level, before digging in.
# import pprint

# chunks = []

# # async for chunk in agent_executor.astream(
# #     {"input": "what's items are located where the cat is hiding?"}
# # ):
# #     chunks.append(chunk)
# #     print("------")
# #     pprint.pprint(chunk, depth=1)

# # chunks[0]["actions"]

# for chunk in chunks:
#     print(chunk["messages"])

# # async for chunk in agent_executor.astream(
# #     {"input": "what's items are located where the cat is hiding?"}
# # ):
# #     # Agent Action
# #     if "actions" in chunk:
# #         for action in chunk["actions"]:
# #             print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
# #     # Observation
# #     elif "steps" in chunk:
# #         for step in chunk["steps"]:
# #             print(f"Tool Result: `{step.observation}`")
# #     # Final result
# #     elif "output" in chunk:
# #         print(f'Final Output: {chunk["output"]}')
# #     else:
# #         raise ValueError()
# #     print("---")

# @tool
# async def get_items(place: str, callbacks: Callbacks) -> str:  # <--- Accept callbacks
#     """Use this tool to look up which items are in the given place."""
#     template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "human",
#                 "Can you tell me what kind of items i might find in the following place: '{place}'. "
#                 "List at least 3 such items separating them by a comma. And include a brief description of each item..",
#             )
#         ]
#     )
#     chain = template | model.with_config(
#         {
#             "run_name": "Get Items LLM",
#             "tags": ["tool_llm"],
#             "callbacks": callbacks,  # <-- Propagate callbacks
#         }
#     )
#     chunks = [chunk async for chunk in chain.astream({"place": place})]
#     return "".join(chunk.content for chunk in chunks)

# # Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# # print(prompt.messages) -- to see the prompt
# tools = [get_items, where_cat_is_hiding]
# agent = create_openai_tools_agent(
#     model.with_config({"tags": ["agent_llm"]}), tools, prompt
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
#     {"run_name": "Agent"}
# )

# async def run_agent():
#     async for event in agent_executor.astream_events(
#         {"input": "where is the cat hiding? what items are in that location?"},
#         version="v1",
#     ):
#         kind = event["event"]
#         if kind == "on_chain_start":
#             if event["name"] == "Agent":  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print(f"Starting agent: {event['name']} with input: {event['data'].get('input')}")
#         elif kind == "on_chain_end":
#             if event["name"] == "Agent":  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
#                 print()
#                 print("--")
#                 print(f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}")
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 # Empty content in the context of OpenAI means
#                 # that the model is asking for a tool to be invoked.
#                 # So we only print non-empty content
#                 print(content, end="|")
#         elif kind == "on_tool_start":
#             print("--")
#             print(f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}")
#         elif kind == "on_tool_end":
#             print(f"Done tool: {event['name']}")
#             print(f"Tool output was: {event['data'].get('output')}")
#             print("--")