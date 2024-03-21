import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_openai import ChatOpenAI

# Set your environment variables using os.environ
os.environ["GITHUB_APP_ID"] = "859649"
os.environ["GITHUB_APP_PRIVATE_KEY"] = "GitHub.pem"
os.environ["GITHUB_REPOSITORY"] = "Chronos-AI-Travel/fake-reseller"
# os.environ["GITHUB_BRANCH"] = "bot-branch-name"
os.environ["GITHUB_BASE_BRANCH"] = "main"

# This example also requires an OpenAI API key
# os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
tools = toolkit.get_tools()

# STRUCTURED_CHAT includes args_schema for each tool, helps tool args parsing errors.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
print("Available tools:")
for tool in tools:
    print("\t" + tool.name)

from langchain.tools.render import render_text_description_and_args

print(render_text_description_and_args(tools))
