"""Integration Agent"""

# Imports
from typing import List, Optional
import os
import httpx
import base64
import json
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
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import asyncio

# Env Variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
session_store = {}

# Firestore
cred = credentials.Certificate("firebase_service_account.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

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


# Fetch File Content
async def fetch_file_content(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        # Assuming the response is JSON and contains a 'content' field encoded in base64
        content_data = json.loads(response.text)
        if (
            "content" in content_data
            and "encoding" in content_data
            and content_data["encoding"] == "base64"
        ):
            # Decode the base64 content
            decoded_content = base64.b64decode(content_data["content"])
            # Convert bytes to string assuming UTF-8 encoding
            return decoded_content.decode("utf-8")
        else:
            # Return an empty string or some error message if 'content' or 'encoding' is not found
            return "Content not found or not in base64 encoding."


# Create Documents
def create_document_for_file(file_name, file_content, llm_response):
    db = firestore.client()
    doc_ref = db.collection("projectFiles").document()
    doc_ref.set(
        {
            "name": file_name,
            "content": file_content,
            "llm_response": llm_response,
            "createdAt": datetime.now(),
        }
    )
    print(f"Document created for file: {file_name} with LLM response")


# Schema
class AgentInvokeRequest(BaseModel):
    input: str = ""
    session_id: str
    docslink: str
    repo: str
    project: str
    suggested_files: Optional[List[str]] = None
    suggested_file_urls: Optional[List[str]] = None
    suggested_file_paths: Optional[List[str]] = None
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


# Agent Route
@app.post("/agent/invoke")
async def agent_invoke(request: AgentInvokeRequest):
    """The Agent"""
    session_id = request.session_id
    project_id = request.project
    db = firestore.client()
    session_data = session_store.get(session_id, {"step": 1})
    documents = create_loader(request.docslink)
    tools = create_tools(documents)
    suggested_files = request.suggested_files
    suggested_file_urls = request.suggested_file_urls
    suggested_file_paths = request.suggested_file_paths
    github_file_contents = await asyncio.gather(
        *[fetch_file_content(url) for url in suggested_file_urls]
    )
    concatenated_github_file_contents = "\n\n---\n\n".join(github_file_contents)
    sanitised_github_file_contents = concatenated_github_file_contents.replace(
        "{", "{{"
    ).replace("}", "}}")

    if session_data["step"] == 1:
        print("Entering Step 1: Starting doc review...")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert travel API integration developer, code specialist",
                ),
                (
                    "user",
                    f"1. Review the API provider docs here: {request.docslink}."
                    "2. Define the Payload required for the request explicitly."
                    "3. Define the clearest error handling for this provider."
                    "Be concise, not verbose.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "docslink": request.docslink,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        docReview_response = response.get(
            "output", "No backend endpoint action performed."
        )
        print(f"docReview_response: {docReview_response}")

        document_name = "integrationStrategy.txt"
        strategy_content = docReview_response
        doc_ref = db.collection("projectFiles").document(document_name)
        doc_ref.set(
            {
                "name": document_name,
                "createdAt": datetime.now(),
                "project": db.collection("projects").document(project_id),
                "code": strategy_content,
            }
        )

        print(f"Document {document_name} created/updated with integration strategy.")

        session_store[session_id] = {
            "step": 2,
            "suggested_files": suggested_files,
            "docReview_response": docReview_response,
        }

        formatted_docReview_response = format_response(docReview_response)
        return {
            "step": 1,
            "message": "Doc review generated",
            "output": formatted_docReview_response,
        }

    elif session_data["step"] == 2:
        print("Entering Step 2: Generating Backend Endpoints...")
        docReview_response = session_data.get("docReview_response", "")
        sanitized_docReview_response = docReview_response.replace("{", "{{").replace(
            "}", "}}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert travel API integration developer, code specialist",
                ),
                (
                    "user",
                    f"1. Review the API provider docs here: {request.docslink}, and the integration plan here: {sanitized_docReview_response}. Pay attention to what is needed in order to integrate with the providers content via API."
                    "Figure out what the backend function should be to make the integration work. It will need to be a route."
                    "Write the python script for the backend that will call the API provider based on their docs."
                    "Ensure you handle allow all CORS."
                    "Ensure you print the response."
                    "Use a flask app that will host this backend locally on port 5000."
                    "Include thorough error handling."
                    "Only return 2 sections in the response: The code, the commands to install any required dependencies."
                    "Comment out any non-code in your response.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "docslink": request.docslink,
            "sanitized_docReview_response": sanitized_docReview_response,
            # "sanitised_github_file_contents": sanitised_github_file_contents,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        backend_endpoint_response = response.get(
            "output", "No backend endpoint action performed."
        )
        print(f"Backend Endpoint result: {backend_endpoint_response}")

        file_created = False
        for file_name in suggested_files:
            if file_name.endswith(".py"):
                doc_ref = db.collection("projectFiles").document(file_name)
                doc_ref.set(
                    {
                        "name": file_name,
                        "createdAt": datetime.now(),
                        "project": db.collection("projects").document(project_id),
                        "code": backend_endpoint_response,
                    }
                )
                print(
                    f"Document created/updated for file: {file_name} with backend endpoint code."
                )
                file_created = True
                break
        if not file_created:
            default_file_name = "app.py"
            doc_ref = db.collection("projectFiles").document(default_file_name)
            doc_ref.set(
                {
                    "name": default_file_name,
                    "createdAt": datetime.now(),
                    "project": db.collection("projects").document(project_id),
                    "code": backend_endpoint_response,
                }
            )
            print(
                f"Default document created for file: {default_file_name} with backend endpoint code."
            )

        session_store[session_id] = {
            "step": 3,
            "suggested_files": suggested_files,
            "backend_endpoint_response": backend_endpoint_response,
            "sanitized_docReview_response": sanitized_docReview_response,
        }

        formatted_backend_response = format_response(backend_endpoint_response)
        return {
            "step": 2,
            "message": "Backend endpoints generated",
            "output": formatted_backend_response,
        }

    elif session_data["step"] == 3:
        print("Entering Step 3: Calculating required functions...")
        backend_endpoint_response = session_data.get("backend_endpoint_response", "")
        sanitized_docReview_response = session_data.get(
            "sanitized_docReview_response", ""
        )
        sanitized_backend_endpoint_response = backend_endpoint_response.replace(
            "{", "{{"
        ).replace("}", "}}")
        print(f"sanitized_docReview_response: {sanitized_docReview_response}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert travel API integration developer, code specialist",
                ),
                (
                    "user",
                    f"1. Review the API provider documentation at {request.docslink}\n."
                    f"2. See the required request payload here here: {sanitized_docReview_response}. This should be factored into the frontend function you create."
                    f"2. Review the backend endpoint we just created for this project here: {sanitized_backend_endpoint_response}."
                    "3. Reply to me only the required frontend function that will correctly call the backend endpoint to make the integration work. assume the backend will be hosted on on http://localhost:5000/"
                    "Only return React code, no explanations, no headers or non-code text. Use fetch instead of axios",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "docslink": request.docslink,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitized_docReview_response": sanitized_docReview_response,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        frontend_function_response = response.get("output", "No action performed.")

        for index, file_name in enumerate(suggested_files):
            file_path = suggested_file_paths[index]  # Get the corresponding file path
            doc_ref = db.collection("projectFiles").document(file_name)
            doc_ref.set(
                {
                    "name": file_name,
                    "createdAt": datetime.now(),
                    "project": db.collection("projects").document(project_id),
                    "code": frontend_function_response,
                    "repoPath": file_path,
                }
            )
            print(
                f"Document created/updated for file: {file_name} with path: {file_path}"
            )

        session_store[session_id] = {
            "step": 4,
            "suggested_files": suggested_files,
            "frontend_function_response": frontend_function_response,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitized_docReview_response": sanitized_docReview_response,
        }

        formatted_agent_response = format_response(frontend_function_response)
        return {
            "step": 3,
            "message": "Refactoring performed on suggested files",
            "output": formatted_agent_response,
        }

    elif session_data["step"] == 4:
        print("Entering Step 4: Creating or Updating UI Components...")
        frontend_function_response = session_data.get("frontend_function_response", "")
        sanitised_frontend_function_response = frontend_function_response.replace(
            "{", "{{"
        ).replace("}", "}}")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        sanitized_docReview_response = session_data.get(
            "sanitized_docReview_response", ""
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert travel API integration developer, code specialist",
                ),
                (
                    "user",
                    "Context: We now need to create the frontend UI elements that will bring the API integration to life."
                    f"1. Review the frontend functions we have made for the integration to the backend {sanitised_frontend_function_response}, These need to be aligned to the UI you are generating now."
                    f"2. Review the payload that has been defined and create the UI based on this too: {sanitized_docReview_response}."
                    "Generate the code for the UI components should be to make the integration work in the UI, request fields and response fields if possible."
                    "Only return React code, no explanation.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "suggested_files": suggested_files,
            "sanitised_frontend_function_response": sanitised_frontend_function_response,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitized_docReview_response": sanitized_docReview_response,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        ui_response = response.get("output", "No UI update action performed.")

        for file_name in suggested_files:
            doc_ref = db.collection("projectFiles").document(file_name)

            doc = doc_ref.get()
            existing_code = doc.to_dict().get("code", "") if doc.exists else ""
            updated_code = existing_code + "\n\n" + ui_response

            doc_ref.update({"code": updated_code})
            print(f"Document for file: {file_name} updated with UI components")

        session_store[session_id] = {
            "step": 5,
            "suggested_files": suggested_files,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitised_frontend_function_response": sanitised_frontend_function_response,
        }

        formatted_ui_response = format_response(ui_response)
        return {
            "step": 4,
            "message": "UI components created or updated",
            "output": formatted_ui_response,
        }

    elif session_data["step"] == 5:
        print("Entering Step 5: API Key section...")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        sanitised_frontend_function_response = session_data.get(
            "sanitised_frontend_function_response", ""
        )

        step_5_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert travel API integration developer."
                    f"1. Search the API providers link: {request.docslink} and learn their process for getting and using the API key."
                    f"2. Provide the steps for me to get and add the API key in my project in a list format. I'm only concerned about the actual API key, nothing else.",
                ),
                (
                    "user",
                    "You are an expert travel API integration developer."
                    f"1. Search the API providers link: {request.docslink} and learn their process for getting and using the API key."
                    f"2. Provide the steps for me to get and add the API key in my project in a list format. I'm only concerned about the actual API key, nothing else."
                    "Include full URLs if they are available for the steps.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "docslink": request.docslink,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=step_5_prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        backend_apiKey_result = response.get(
            "output", "No backend endpoint action performed."
        )
        print(f"Backend Endpoint result: {backend_apiKey_result}")

        session_store[session_id] = {
            "step": 6,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitised_frontend_function_response": sanitised_frontend_function_response,
        }

        steps_list = backend_apiKey_result.split("\n")

        return {
            "step": 5,
            "message": "API Key info sent",
            "output": steps_list,
        }

    elif session_data["step"] == 6:
        print("Entering Step 6: Creating Integration Tests...")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        sanitised_frontend_function_response = session_data.get(
            "sanitised_frontend_function_response", ""
        )
        docslink = request.docslink

        step_6_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Travel API Integrator focusing on quality assurance. ",
                ),
                (
                    "user",
                    "Your task now is to create backend in the same language as the provided backend code integration tests for the API provider based on the integration requirements identified in the previous steps."
                    "Consider the functionalities proposed for integration and ensure the tests cover these functionalities effectively."
                    "Write the code for the integration tests, nothing else, literally."
                    f"\n\nIntegration Actions from Step 2:\n{sanitised_frontend_function_response}\n"
                    f"\nBackend Endpoint Result from Step 4:\n{sanitized_backend_endpoint_response}\n"
                    f"\nDocumentation Link: {docslink}\n",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "sanitised_frontend_function_response": sanitised_frontend_function_response,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=step_6_prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        integration_tests_result = response.get(
            "output", "No integration tests action performed."
        )
        print(f"Integration Tests result: {integration_tests_result}")

        integration_tests_file_name = "integration_tests.py"
        project_id = request.project
        doc_ref = db.collection("projectFiles").document(integration_tests_file_name)
        doc_ref.set(
            {
                "code": integration_tests_result,
                "createdAt": datetime.now(),
                "name": integration_tests_file_name,
                "project": db.collection("projects").document(project_id),
            }
        )
        print(
            f"Document created for file: {integration_tests_file_name} with integration tests code."
        )

        session_store[session_id] = {
            "step": 7,
            "integration_tests_result": integration_tests_result,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
        }

        formatted_integration_tests_response = format_response(integration_tests_result)
        return {
            "step": 6,
            "message": "Integration tests created",
            "output": formatted_integration_tests_response,
        }

    elif session_data["step"] == 7:
        print("Entering Step 7: Reviewing code for improvements...")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )

        if suggested_files:
            for file_name in suggested_files:
                if file_name.endswith(".js"):
                    doc_ref = db.collection("projectFiles").document(file_name)
                    doc = doc_ref.get()
                    if doc.exists:
                        frontend_generated_code = doc.to_dict().get("code", "")
                    else:
                        frontend_generated_code = (
                            "No code found in the document for the '.js' file."
                        )
                    break
        sanitised_frontend_generated_code = frontend_generated_code.replace(
            "{", "{{"
        ).replace("}", "}}")
        print(f"sanitised_frontend_generated_code: {sanitised_frontend_generated_code}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Travel API Integrator, code generator.",
                ),
                (
                    "user",
                    f"1. Review the API content provider docs at {request.docslink}."
                    f"2. Review the code at {sanitised_frontend_generated_code}."
                    "3. Now turn it into working code."
                    "It should contain the frontend function(s) calling the backend function as defined, the ui components and any necessary boilerplate. Assume the component is called App."
                    "I need to be able to copy and paste this into my file and it should work, so React code only. Assume this frontend code is all in one file, and the backend code with the API call to the provider is in a seperate file."
                    "Comment out any non-code in your response.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "sanitised_frontend_generated_code": sanitised_frontend_generated_code,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "docslink": request.docslink,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        code_review_response = response.get(
            "output", "No impact analysis action performed."
        )
        print(f"Impact Analysis result: {code_review_response}")

        if (
            suggested_files
            and code_review_response != "No impact analysis action performed."
        ):
            doc_ref.update({"code": code_review_response})
            print(f"Document for file: {file_name} updated with new code.")

        session_store[session_id] = {
            "step": 8,
        }

        formatted_code_review_response = format_response(code_review_response)
        return {
            "step": 7,
            "message": "Code review completed",
            "output": formatted_code_review_response,
        }


def format_response(frontend_function_response):
    parts = frontend_function_response.split("\n")
    formatted_response = ""
    code_block_open = False

    for part in parts:
        if part.endswith(".js"):
            if code_block_open:
                formatted_response += "</code></pre>"
            formatted_response += f"<h3>{part}</h3><pre><code>"
            code_block_open = True
        else:
            formatted_response += f"{part}\n"

    if code_block_open:
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
