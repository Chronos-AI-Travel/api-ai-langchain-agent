"""Langchain Duffel Agent"""

from typing import List
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
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
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import asyncio
from concurrent.futures import ThreadPoolExecutor


# logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


# 1. Load Retriever
loader = WebBaseLoader("https://duffel.com/docs/guides/getting-started-with-flights")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Repo Indexing
namespace = "elasticsearch/your_index_name"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)
record_manager.create_schema()
vectorstore = FAISS.from_documents(documents, embeddings)

doc1 = Document(page_content="Example content 1", metadata={"source": "file1.txt"})
doc2 = Document(page_content="Example content 2", metadata={"source": "file2.txt"})

index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)



# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "Duffel_docs",
    "Search for information about integrating with Duffel. For any questions about what code to suggest, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]


# 3. Create Agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Travel API Integrator. Your mission is to integrate the user's files with the Duffel API based on the content of their repository. "
            "1. Review the indexed Duffel docs to understand the API. "
            "2. List out the repository files for me. "
            "3. Analyze the repository files to identify where and how the Duffel API can be integrated. "
            "4. Suggest specific files and code changes for Duffel API integration.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Add CORSMiddleware to the application instance to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 5. Schema
class GithubInfo(BaseModel):
    # access_token: str
    repo: str


class AgentInvokeRequest(BaseModel):
    input: str = ""  # Set a default value to make it optional
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )
    github_info: GithubInfo


# Routes
@app.post("/index/repository")
async def index_repository(repo_info: GithubInfo, background_tasks: BackgroundTasks):
    """
    Endpoint to asynchronously index documents from a specified GitHub repository.
    """
    # Validate the GitHub access token and repo information
    if not repo_info.repo or not access_token:
        raise HTTPException(status_code=400, detail="Missing repository information or access token.")

    # Use background tasks to handle the loading and indexing without blocking the API response
    background_tasks.add_task(load_and_index_documents, repo_info.repo, access_token)

    return {"message": f"Indexing of documents from {repo_info.repo} has been initiated."}

async def load_and_index_documents(repo, access_token):
    def load_documents():
        return github_loader.load()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        github_loader = GithubFileLoader(
            repo=repo,
            access_token=access_token,
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith(".txt")
            or file_path.endswith(".md")
            or file_path.endswith(".js")
            or file_path.endswith(".json"),
        )
        # Run the synchronous load method in a thread pool
        github_documents = await loop.run_in_executor(pool, load_documents)

        # Initialize the embeddings generator
        embeddings_generator = OpenAIEmbeddings()

        # Process each document
        for doc in github_documents:
            # Generate embeddings for each document
            doc_embedding = embeddings_generator.generate(doc.content)  # Adjust this line based on your actual method
            
            # Ensure your index function can handle the document and its embedding appropriately
            index([doc], doc_embedding, record_manager, vectorstore, cleanup="incremental", source_id_key="source")

        print(f"Indexed {len(github_documents)} documents from the repository.")

    # Validate the GitHub access token and repo information
    if not repo_info.repo or not access_token:
        return {"message": "Missing repository information or access token."}

    # Use background tasks to handle the loading and indexing without blocking the API response
    background_tasks.add_task(load_and_index_documents, repo_info.repo, access_token)

    return {"message": f"Indexing of documents from {repo_info.repo} has been initiated."}


@app.post("/agent/invoke")
async def agent_invoke(request: AgentInvokeRequest):
    """Invoke the agent response"""
    # Assuming the documents from both sources are already indexed
    print(f"Received input: {request.input}")

    # Dynamically adjust the search query based on the input or a specific need
    dynamic_search_query = "search query based on request.input or other logic"
    query_results = vectorstore.similarity_search(dynamic_search_query, k=10)
    relevant_docs = [result.metadata["source"] for result in query_results]
    print("Relevant documents:", relevant_docs)

    # Prepare the context with both Duffel docs and GitHub repo content if needed
    # This example assumes you have a way to fetch or reference the content as needed
    combined_context = "Combined context from Duffel docs and GitHub repo documents"

    try:
        context = {
            "input": request.input,
            "chat_history": request.chat_history,
            "combined_context": combined_context,  # Adjusted to use combined context
        }
        response = await agent_executor.ainvoke(context)
        agent_response = response.get("output", "No response generated.")
        print(f"Response from agent: {agent_response}")
    except Exception as e:
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


if __name__ == "__main__":
    uvicorn.run(
        "serve:app", host="localhost", port=8000, log_level="debug", reload=True
    )
