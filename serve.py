"""Integration Agent"""

from typing import List
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GithubFileLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# logging.basicConfig(level=logging.DEBUG)

# Env Variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# App
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Loader Tool
def create_loader(docslink: str):
    loader = WebBaseLoader(docslink)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents

# Tools
def create_tools(documents):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "docs_retriever",
        "Search for information about integrating with a travel provider. For any questions about what code to suggest, you must use this tool!",
    )
    search = TavilySearchResults()
    tools = [retriever_tool, search]
    return tools

# Agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Travel API Integrator. Your mission is to integrate the appropriate files with the provided API docs based on the name of the files loaded from the repository. "
            "1. Here are my repository files."
            "\n\nRepository Files:\n{file_list}\n\nRepository Content:\n{github_file_content}"
            "2. Given the following list of repository files and the documentation link to an API, identify which files are most relevant for integrating the API. Consider synonyms like Stays and Hotels are matches, as is Flights and Air, you make the decision which files are most appropriate."
            "3. Whichever files you think are most appropriate, then write out the new code for that file to complete the integration as much as possible."
            "4. The response format should be: 1st line: file name, remaining response is the code, no other commentary",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Schema
class AgentInvokeRequest(BaseModel):
    input: str = ""
    docslink: str
    repo: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

# Agent Route
@app.post("/agent/invoke")
async def agent_invoke(request: AgentInvokeRequest):
    """Invoke the agent response"""
    print(f"Request body: {request.json()}")
    documents = create_loader(request.docslink)
    tools = create_tools(documents)
    agent = create_openai_functions_agent(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    github_loader = GithubFileLoader(
        repo=request.repo,
        access_token=access_token,
        github_api_url="https://api.github.com",
       file_filter=lambda file_path: (file_path.endswith(".js") 
        and not "config" in file_path 
        and not file_path.endswith("layout.js"))
        or file_path == "package.json",
    )
    github_documents = github_loader.load()
    file_paths = [doc.metadata["path"] for doc in github_documents]
    file_list_str = "\n".join(file_paths)
    if github_documents:
        print("Files loaded from the repository:")
        for doc in github_documents:
            print(doc.metadata["path"])
    else:
        print("No documents were loaded from the repository.")

    github_file_content = "\n".join([doc.page_content for doc in github_documents])
    try:
        context = {
            "input": request.input,
            "chat_history": request.chat_history,
            "github_file_content": github_file_content,
            "file_list": file_list_str,
        }
        response = await agent_executor.ainvoke(context)
        agent_response = response.get("output", "No response generated.")
        print(f"Response from agent: {agent_response}")
    except Exception as e:  # pylint: disable=broad-except
        exception_type = e.__class__.__name__
        print(f"Error processing request: {exception_type}: {e}")
        return {
            "output": f"An error occurred while processing your request: {exception_type}"
        }

    def format_response(agent_response):
        parts = agent_response.split("```")
        formatted_response = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                formatted_response += f"<p>{part}</p>"
            else:
                escaped_code = part.replace("<", "&lt;").replace(">", "&gt;")
                formatted_response += f"<pre><code>{escaped_code}</code></pre>"
        return formatted_response

    formatted_agent_response = format_response(agent_response)
    return {"output": formatted_agent_response}

# Root Route
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Server
if __name__ == "__main__":
    uvicorn.run(
        "serve:app", host="localhost", port=8000, log_level="debug", reload=True
    )
