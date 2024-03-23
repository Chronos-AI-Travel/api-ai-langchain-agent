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

# Env Variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

session_store = {}

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
            "\n\nRepository Files:\n{file_list}\n"
            # "\nRepository Content:\n{github_file_content}"
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
    session_id: str
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
    session_id = request.session_id
    session_data = session_store.get(session_id, {"step": 1})
    if session_data["step"] == 1:
        # Step 1: Analyze docs and suggest files
        documents = create_loader(request.docslink)
        print(f"Documents loaded: {len(documents)} documents")
        tools = create_tools(documents)
        agent = create_openai_functions_agent(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
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
        print(f"GitHub documents loaded: {len(github_documents)} documents")        
        file_paths = [doc.metadata["path"] for doc in github_documents]
        file_list_str = "\n".join(file_paths)
        github_file_content = "\n".join([doc.page_content for doc in github_documents])
        context = {
            "input": 'based on this list of file names, and the docs provided to you in this link, tell me the list of file names which are most likely to be appropriate for integration with these docs. Just provide me with the file names, no other text before or after the file names, literally.',
            "chat_history": request.chat_history,
            "docslink": request.docslink,
            "repo": request.repo,
            # "github_file_content": github_file_content, 
            "file_list": file_list_str, 
        }
        print("Context prepared for agent invocation.")        
        response = await agent_executor.ainvoke(context)
        agent_response = response.get("output", "No response generated.")
        print(f"Response from agent: {agent_response}")
        suggested_files = response.get("output", "No files suggested.")
        print(f"Suggested files: {suggested_files}")    

        suggested_files_list = suggested_files.split('\n')  
        session_store[session_id] = {
            "step": 2,
            "suggested_files": suggested_files_list,  
            "github_file_content": github_file_content,
            "file_list": file_list_str,
            "docslink": request.docslink 

        }
        return {"step": 1, "message": "Files suggested for refactoring", "suggested_files": suggested_files}

    elif session_data["step"] == 2:
        print("Entering Step 2: Refactoring suggested files...")
        # Retrieve necessary data for step 2
        suggested_files = session_data.get("suggested_files", []) 
        documents = create_loader(request.docslink)
        tools = create_tools(documents)
        github_file_content = session_data.get("github_file_content", "")
        print(f"github_file_content: {github_file_content}")
        file_list_str = session_data.get("file_list", "")
        docslink = session_data.get("docslink", "")
        file_contents = github_file_content.split('\n\n---\n\n')
        file_paths = file_list_str.split('\n')
        filtered_content = "\n\n---\n\n".join(content for file_path, content in zip(file_paths, file_contents) if file_path in suggested_files)
        context = {
            "input": f"Based on the provided documentation found at {docslink}, refactor the code of the files listed below to result in integration with the provider. The files suggested for refactoring are: {', '.join(suggested_files)}. The content of these files is as follows:\n\n---\n\n{filtered_content}\n\n---\n\nPlease provide the refactored code for each file.",
            "chat_history": request.chat_history,
            "suggested_files": suggested_files,
            "repo": request.repo,
            "github_file_content": filtered_content,  
            "file_list": file_list_str,
            "docslink": docslink,
        }
        agent = create_openai_functions_agent(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        action_result = response.get("output", "No action performed.")
        print(f"Refactoring result: {action_result}")

    del session_store[session_id]
    print("Session data cleaned up after step 2 completion.")

    formatted_agent_response = format_response(action_result)
    return {"step": 2, "message": "Refactoring performed on suggested files", "output": formatted_agent_response}
        
def format_response(action_result):
    # Split the action_result by new lines to handle multiple files or sections
    parts = action_result.split("\n")
    formatted_response = ""
    code_block_open = False

    for part in parts:
        # Check if the line indicates the start of a new file or code block
        if part.endswith(".js"):
            if code_block_open:
                # Close previous code block
                formatted_response += "</code></pre>"
            # Add the file name as a header and open a new code block
            formatted_response += f"<h3>{part}</h3><pre><code>"
            code_block_open = True
        else:
            # Add the code line or content to the current code block
            formatted_response += f"{part}\n"

    if code_block_open:
        # Close the last code block if open
        formatted_response += "</code></pre>"

    return formatted_response

# Root Route
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Server
if __name__ == "__main__":
    uvicorn.run(
        "serve:app", host="localhost", port=8000, log_level="debug", reload=True
    )
