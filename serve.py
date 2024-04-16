"""Integration Agent"""

# Imports
from typing import List, Optional
import os
import httpx
import base64
import json
import uvicorn
import re
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


# Fetch Capability Data
async def fetch_capability_data(db, doc_path):
    doc_ref = db.document(doc_path)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        print(f"No document found for path: {doc_path}")
        return None


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
    frontendFramework: str
    backendFramework: str
    frontend_file_names: Optional[List[str]] = None
    frontend_file_urls: Optional[List[str]] = None
    frontend_file_paths: Optional[List[str]] = None
    backend_file_names: Optional[List[str]] = None
    backend_file_urls: Optional[List[str]] = None
    backend_file_paths: Optional[List[str]] = None
    capabilityRefs: Optional[List[str]] = None
    userRequestFields: Optional[List[str]] = None
    userResponseFields: Optional[List[str]] = None
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
    session_data = session_store.get(session_id, {"step": 2})
    documents = create_loader(request.docslink)
    tools = create_tools(documents)

    # Ensure all file-related lists are iterable
    frontend_file_names = request.frontend_file_names or []
    frontend_file_urls = request.frontend_file_urls or []
    frontend_file_paths = request.frontend_file_paths or []
    backend_file_names = request.backend_file_names or []
    backend_file_urls = request.backend_file_urls or []
    backend_file_paths = request.backend_file_paths or []
    frontend_file_contents = await asyncio.gather(
        *[fetch_file_content(url) for url in frontend_file_urls]
    )
    sanitized_frontend_contents_by_url = {
        url: content.replace("{", "{{").replace("}", "}}")
        for url, content in zip(frontend_file_urls, frontend_file_contents)
    }
    concatenated_sanitized_frontend_contents = "\n".join(
        [
            content.replace("{", "{{").replace("}", "}}")
            for content in frontend_file_contents
        ]
    )

    # Fetch and sanitize contents for backend files
    backend_file_contents = await asyncio.gather(
        *[fetch_file_content(url) for url in backend_file_urls]
    )
    sanitized_backend_contents_by_url = {
        url: content.replace("{", "{{").replace("}", "}}")
        for url, content in zip(backend_file_urls, backend_file_contents)
    }
    concatenated_sanitized_backend_contents = "\n".join(
        [
            content.replace("{", "{{").replace("}", "}}")
            for content in backend_file_contents
        ]
    )

    db = firestore.client()
    capabilities_names = []
    capabilities_endPoints = []
    capabilities_headers = []
    capabilities_method = []
    capabilities_routeName = []
    capabilities_errorBody = []
    capabilities_requestBody = []
    capabilities_responseBody = []
    capabilities_responseGuidance = []
    capabilities_requestGuidance = []
    sanitized_capabilities_headers = []
    sanitized_capabilities_errorBody = []
    sanitized_capabilities_requestBody = []
    sanitized_capabilities_responseBody = []
    sanitized_capabilities_responseGuidance = []
    sanitized_capabilities_requestGuidance = []

    # Fetch capability data
    db = firestore.client()
    if request.capabilityRefs:
        capability_docs = await asyncio.gather(
            *[fetch_capability_data(db, path) for path in request.capabilityRefs]
        )
        for doc_data in capability_docs:
            if doc_data:
                capabilities_names.append(doc_data.get("name", "No name"))
                capabilities_endPoints.append(doc_data.get("endPoint", "No endPoint"))
                capabilities_headers.append(doc_data.get("headers", "No headers"))
                capabilities_routeName.append(doc_data.get("routeName", "No routeName"))
                capabilities_method.append(doc_data.get("method", "No method"))
                capabilities_errorBody.append(doc_data.get("errorBody", "No errorBody"))
                capabilities_requestBody.append(
                    doc_data.get("requestBody", "No requestBody")
                )
                capabilities_responseBody.append(
                    doc_data.get("responseBody", "No responseBody")
                )
                capabilities_responseGuidance.append(
                    doc_data.get("responseGuidance", "No responseGuidance")
                )
                capabilities_requestGuidance.append(
                    doc_data.get("requestGuidance", "No requestGuidance")
                )
                sanitized_capabilities_errorBody = [
                    errorBody.replace("{", "{{").replace("}", "}}")
                    for errorBody in capabilities_errorBody
                ]
                sanitized_capabilities_headers = [
                    headers.replace("{", "{{").replace("}", "}}")
                    for headers in capabilities_headers
                ]
                sanitized_capabilities_requestBody = [
                    requestBody.replace("{", "{{").replace("}", "}}")
                    for requestBody in capabilities_requestBody
                ]
                sanitized_capabilities_responseBody = [
                    responseBody.replace("{", "{{").replace("}", "}}")
                    for responseBody in capabilities_responseBody
                ]
                sanitized_capabilities_responseGuidance = [
                    responseGuidance.replace("{", "{{").replace("}", "}}")
                    for responseGuidance in capabilities_responseGuidance
                ]
                sanitized_capabilities_requestGuidance = [
                    requestGuidance.replace("{", "{{").replace("}", "}}")
                    for requestGuidance in capabilities_requestGuidance
                ]

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
                    "2. Return the Payload / request body schema object required for the request, only include the required body parameters and the data structure. Note on each field when it is required."
                    "3. Also return the Response data object and its data structure.",
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

        print(f"Document {document_name} created/updated with doc review.")

        session_store[session_id] = {
            "step": 2,
            # "suggested_files": suggested_files,
            "docReview_response": docReview_response,
        }

        formatted_docReview_response = format_response(docReview_response)
        return {
            "step": 1,
            "message": "Doc review generated",
            "output": formatted_docReview_response,
        }

    elif session_data["step"] == 2:
        print(f"capabilities_routeName: {capabilities_routeName}")
        print(f"sanitized_capabilities_headers: {sanitized_capabilities_headers}")
        print(f"capabilities_endPoints: {capabilities_endPoints}")
        print(f"capabilities_method: {capabilities_method}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert travel API integration developer, your mission is to generate a backend route in {request.backendFramework}.",
                ),
                (
                    "user",
                    "# Start your response with a comment and end your response with a comment.\n"
                    "Create a backend route that acts as an API proxy."
                    "Do not use the provider docs, only use the data provided below for this request:"
                    f"Route name: {capabilities_routeName}."
                    f"Do not hardcode the payload."
                    f"Method: {capabilities_method}."
                    f"Headers: {sanitized_capabilities_headers}."
                    f"Endpoint url: {capabilities_endPoints}."
                    f"Consider the error logging if required: \n{sanitized_capabilities_errorBody}."
                    "Handle the response."
                    f"Integrate the new code into the existing code found here: {concatenated_sanitized_backend_contents}"
                    "Integrate new code without altering or removing existing code."
                    "Ensure you handle allow all CORS."
                    f"Use a {request.backendFramework} app that will host this backend locally on port 5000."
                    "Add print statements for errors and the response."
                    "Be concise, only respond with the code.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "capabilities_endPoints": capabilities_endPoints,
            "sanitized_capabilities_headers": sanitized_capabilities_headers,
            "capabilities_routeName": capabilities_routeName,
            "sanitized_capabilities_errorBody": sanitized_capabilities_errorBody,
            "concatenated_sanitized_backend_contents": concatenated_sanitized_backend_contents,
            "request.backendFramework": request.backendFramework,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        backend_endpoint_response = response.get(
            "output", "No backend endpoint action performed."
        )
        formatted_backend_response = format_response(backend_endpoint_response)

        file_created = False
        for index, file_name in enumerate(backend_file_names):
            if file_name.endswith(".py"):
                file_path = backend_file_paths[index]  # Example way to get the repoPath
                doc_ref = db.collection("projectFiles").document(file_name)
                doc_ref.set(
                    {
                        "name": file_name,
                        "createdAt": datetime.now(),
                        "project": db.collection("projects").document(project_id),
                        "code": formatted_backend_response,  # Updated to use formatted response
                        "repoPath": file_path,  # Include the repoPath
                    }
                )
                print(
                    f"Document created/updated for file: {file_name} with backend endpoint code and repoPath."
                )
                file_created = True
                break

        if not file_created:
            default_file_name = "app.py"
            default_file_path = (
                "path/to/default/app.py"  # Example path for the default file
            )
            doc_ref = db.collection("projectFiles").document(default_file_name)
            doc_ref.set(
                {
                    "name": default_file_name,
                    "createdAt": datetime.now(),
                    "project": db.collection("projects").document(project_id),
                    "code": formatted_backend_response,  # Updated to use formatted response
                    "repoPath": default_file_path,  # Include the repoPath for the default file
                }
            )
            print(
                f"Default document created for file: {default_file_name} with backend endpoint code and repoPath."
            )
            print(
                f"Default document created for file: {default_file_name} with backend endpoint code."
            )

        session_store[session_id] = {
            "step": 3,
            "backend_file_names": backend_file_names,
            "backend_endpoint_response": backend_endpoint_response,
        }

        formatted_backend_response = format_response(backend_endpoint_response)
        return {
            "step": 2,
            "message": "Backend endpoints generated",
            "output": formatted_backend_response,
        }

    elif session_data["step"] == 3:
        print("Entering Step 3: Creating or Updating Request UI Elements...")
        print(f"request.userRequestFields: {request.userRequestFields}")
        print(
            f"sanitized_capabilities_requestGuidance: {sanitized_capabilities_requestGuidance}"
        )
        backend_endpoint_response = session_data.get("backend_endpoint_response", "")
        sanitized_backend_endpoint_response = backend_endpoint_response.replace(
            "{", "{{"
        ).replace("}", "}}")

        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert {request.frontendFramework} developer. Your task is to generate UI elements for constructing an API request payload.",
                ),
                (
                    "user",
                    "// Start your response with a comment and end your response with a comment.\n"
                    f"Create {request.frontendFramework} UI elements for the request part of the API integration, such as form fields (e.g. buttons, text fields, inputs, date pickers for date fields, dropdowns for select)."
                    "Do not use the provider docs, only use the data provided below for this request:"
                    f"Constructing the payload according to the API's expected structure: {sanitized_capabilities_requestBody}."
                    f"Create only the following request fields: {sanitized_capabilities_requestGuidance} {request.userRequestFields}."
                    f"Integrate the new code into the existing code, without altering or removing existing code, found here: {concatenated_sanitized_frontend_contents}, "
                    "Avoid using placeholders that might suggest removing existing code."
                    "Keep all frontend code in a single component."
                    "Create the required state fields."
                    "Minimise the need for package installations."
                    "Keep existing imports and create any required imports."
                    f"Only return {request.frontendFramework} code. Ensure the solution is complete and accurate.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            # "frontend_file_names": frontend_file_names,
            "sanitized_capabilities_requestBody": sanitized_capabilities_requestBody,
            "sanitized_capabilities_requestGuidance": sanitized_capabilities_requestGuidance,
            "concatenated_sanitized_frontend_contents": concatenated_sanitized_frontend_contents,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        response_ui_response = response.get("output", "No UI update action performed.")
        formatted_ui_response = format_response(response_ui_response)

        for index, file_name in enumerate(frontend_file_names):
            file_path = frontend_file_paths[index]
            doc_ref = db.collection("projectFiles").document(file_name)
            doc_ref.set(
                {
                    "name": file_name,
                    "createdAt": datetime.now(),
                    "project": db.collection("projects").document(project_id),
                    "code": formatted_ui_response,
                    "repoPath": file_path,
                }
            )
            print(
                f"Document created/updated for file: {file_name} with path: {file_path} and formatted UI response."
            )

        session_store[session_id] = {
            "step": 4,
            "frontend_file_names": frontend_file_names,
            "response_ui_response": response_ui_response,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            # "sanitised_frontend_function_response": sanitised_frontend_function_response,
            # "sanitized_docReview_response": sanitized_docReview_response,
        }

        return {
            "step": 3,
            "message": "UI components created or updated",
            "output": formatted_ui_response,
        }

    elif session_data["step"] == 4:
        print("Entering Step 4: Creating or Updating UI Response Elements...")
        print(f"request.userResponseFields: {request.userResponseFields}")
        if frontend_file_names:
            for file_name in frontend_file_names:
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
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        response_ui_response = session_data.get("response_ui_response", "")
        sanitized_response_ui_response = response_ui_response.replace(
            "{", "{{"
        ).replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert {request.frontendFramework} developer. Your task now is to create UI elements for displaying API response data.",
                ),
                (
                    "user",
                    "// Start your response with a comment and end your response with a comment.\n"
                    f"Create for me frontend {request.frontendFramework} UI elements for the response part of the API integration, such as form fields (e.g. text, tables, lists, card etc)."
                    "Do not use the provider docs, only use the data provided below for this request:"
                    f"Structure the response according to the response data object: {sanitized_capabilities_responseBody}."
                    f"Follow this advice to structure the response properly: {sanitized_capabilities_responseGuidance} also display this data: {request.userResponseFields}."
                    f"Integrate the new code into the existing code found here: {sanitised_frontend_generated_code}"
                    "Integrate new code without altering or removing existing code, you must add to the existing code and respond with the full code."
                    "Avoid using placeholders that might suggest removing existing code."
                    "Keep all frontend code in a single component."
                    "No dummy data. Do not create the API call handler."
                    f"Only return {request.frontendFramework} code. Ensure the solution is complete and accurate.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "sanitized_capabilities_responseBody": sanitized_capabilities_responseBody,
            "sanitized_capabilities_responseGuidance": sanitized_capabilities_responseGuidance,
            "sanitised_frontend_generated_code": sanitised_frontend_generated_code,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        request_ui_response = response.get("output", "No UI update action performed.")
        formatted_request_ui_response = format_response(request_ui_response)

        for index, file_name in enumerate(frontend_file_names):
            file_path = frontend_file_paths[index]
            doc_ref = db.collection("projectFiles").document(file_name)
            doc_ref.set(
                {
                    "name": file_name,
                    "createdAt": datetime.now(),
                    "project": db.collection("projects").document(project_id),
                    "code": formatted_request_ui_response,
                    "repoPath": file_path,
                }
            )
            print(
                f"Document created/updated for file: {file_name} with path: {file_path} and formatted UI response."
            )

        session_store[session_id] = {
            "step": 5,
            "frontend_file_names": frontend_file_names,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "formatted_request_ui_response": formatted_request_ui_response,
            "sanitized_response_ui_response": sanitized_response_ui_response,
            "sanitized_capabilities_responseBody": sanitized_capabilities_responseBody,
        }

        return {
            "step": 4,
            "message": "UI Response components created or updated",
            "output": formatted_request_ui_response,
        }

    elif session_data["step"] == 5:
        print("Entering Step 5: Calculating API request-response handler...")
        if frontend_file_names:
            for file_name in frontend_file_names:
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
        request_ui_response = session_data.get("request_ui_response", "")
        formatted_request_ui_response = session_data.get(
            "formatted_request_ui_response", ""
        )
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )

        sanitized_formatted_request_ui_response = formatted_request_ui_response.replace(
            "{", "{{"
        ).replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert {request.frontendFramework} developer. Your final task is to integrate the UI elements with an API request-response handler.",
                ),
                (
                    "user",
                    "// Note: Start your response with a comment (using '//') and also end your response with a comment (using '//').\n"
                    f"Generate {request.frontendFramework} code for the the frontend API request-response handler that will handle the request and response to the backend we have defined here: {sanitized_backend_endpoint_response}."
                    f"Route name: {capabilities_routeName}"
                    f"Method: {capabilities_method}"
                    "Assume the backend will be hosted on on http://localhost:5000/."
                    "Do not use the provider docs, only use the data provided below for this request:"
                    f"See the UI fields we have here and write the API request-response handler to handle them: {sanitised_frontend_generated_code}."
                    f"Request query parameters: {sanitized_capabilities_requestBody}"
                    f"Response structure: {sanitized_capabilities_responseBody}"
                    "Return to me the existing code, plus the API request-response handler component."
                    "Do not use any dummy data."
                    "Integrate new code without altering or removing existing code, you must add to the existing code and respond with the full code."
                    "Avoid using placeholders that might suggest removing existing code."
                    "Keep all frontend code in a single component."
                    "If nothing needs changing, then dont change anything and return the full existing code."
                    f"Only return {request.frontendFramework} code. Use Fetch not axios. Ensure the solution is complete and accurate.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "sanitized_capabilities_requestBody": sanitized_capabilities_requestBody,
            "sanitised_frontend_generated_code": sanitised_frontend_generated_code,
            "capabilities_routeName": capabilities_routeName,
            "sanitized_capabilities_responseBody": sanitized_capabilities_responseBody,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        frontend_function_response = response.get("output", "No action performed.")

        # Format the frontend_function_response before saving it
        formatted_frontend_function_response = format_response(
            frontend_function_response
        )

        for file_name in frontend_file_names:
            doc_ref = db.collection("projectFiles").document(file_name)
            # Use formatted_frontend_function_response here
            doc_ref.update({"code": formatted_frontend_function_response})
            print(
                f"Document for file: {file_name} updated with formatted frontend function"
            )

        session_store[session_id] = {
            "step": 7,
            "suggested_files": frontend_file_names,
            "frontend_function_response": frontend_function_response,  # Original response
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
        }

        formatted_agent_response = format_response(frontend_function_response)
        return {
            "step": 5,
            "message": "Refactoring performed on suggested files",
            "output": formatted_agent_response,
        }

    elif session_data["step"] == 6:
        print("Entering Step 6: Calculating API request-response handler...")
        if frontend_file_names:
            for file_name in frontend_file_names:
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
        request_ui_response = session_data.get("request_ui_response", "")
        formatted_request_ui_response = session_data.get(
            "formatted_request_ui_response", ""
        )
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )

        sanitized_formatted_request_ui_response = formatted_request_ui_response.replace(
            "{", "{{"
        ).replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"You are an expert travel API integration developer, your mission is to generate the Response part of a frontend API request-response handler in {request.frontendFramework}.",
                ),
                (
                    "user",
                    "// Note: Start your response with a comment (using '//') and also end your response with a comment (using '//').\n"
                    "Do not use the provider docs, only use the data provided below for this code:"
                    f"See the the existing code we have here and write the API request-response handler to handle the fields: {sanitised_frontend_generated_code}."
                    f"Structure it according to the response object structure: {capabilities_responseBody}."
                    "Return to me the code updated with the frontend API request-response handler component."
                    "Do not use any dummy data."
                    "Integrate new code without altering or removing existing code, you must add to the existing code and response with the full code."
                    "Avoid using placeholders that might suggest removing existing code."
                    "Keep all frontend code in a single component."
                    f"Only return {request.frontendFramework} code. Use fetch instead of axios. Be concise.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitized_capabilities_errorBody": sanitized_capabilities_errorBody,
            "sanitized_capabilities_requestBody": sanitized_capabilities_requestBody,
            "sanitized_capabilities_responseBody": sanitized_capabilities_responseBody,
            "sanitized_capabilities_responseGuidance": sanitized_capabilities_responseGuidance,
            "sanitized_formatted_request_ui_response": sanitized_formatted_request_ui_response,
            "sanitised_frontend_generated_code": sanitised_frontend_generated_code,
            "capabilities_routeName": capabilities_routeName,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = await agent_executor.ainvoke(context)
        frontend_function_response = response.get("output", "No action performed.")

        # Format the frontend_function_response before saving it
        formatted_frontend_function_response = format_response(
            frontend_function_response
        )

        for file_name in frontend_file_names:
            doc_ref = db.collection("projectFiles").document(file_name)
            # Use formatted_frontend_function_response here
            doc_ref.update({"code": formatted_frontend_function_response})
            print(
                f"Document for file: {file_name} updated with formatted frontend function"
            )

        session_store[session_id] = {
            "step": 7,
            "frontend_file_names": frontend_file_names,
            "frontend_function_response": frontend_function_response,  # Original response
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
        }

        formatted_agent_response = format_response(frontend_function_response)
        return {
            "step": 6,
            "message": "Refactoring performed on suggested files",
            "output": formatted_agent_response,
        }

    elif session_data["step"] == 7:
        print("Entering Step 7: Creating Integration Tests...")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        sanitised_frontend_function_response = session_data.get(
            "sanitised_frontend_function_response", ""
        )
        sanitized_docReview_response = session_data.get(
            "sanitized_docReview_response", ""
        )
        docslink = request.docslink

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Travel API Integrator focusing on quality assurance. ",
                ),
                (
                    "user",
                    "// Note: Start your response with a comment (using '//') and also end your response with a comment (using '//').\n"
                    f"Your task now is to create backend in {request.backendFramework} for the API provider based on the integration requirements identified in the previous steps."
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
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            tools=tools,
            prompt=prompt,
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
            "step": 8,
            "integration_tests_result": integration_tests_result,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
            "sanitized_docReview_response": sanitized_docReview_response,
        }

        formatted_integration_tests_response = format_response(integration_tests_result)
        return {
            "step": 7,
            "message": "Integration tests created",
            "output": formatted_integration_tests_response,
        }

    elif session_data["step"] == 8:
        print("Entering Step 8: Documentation...")
        sanitized_backend_endpoint_response = session_data.get(
            "sanitized_backend_endpoint_response", ""
        )
        sanitised_frontend_generated_code = session_data.get(
            "sanitised_frontend_generated_code", ""
        )
        # sanitized_docReview_response = session_data.get(
        #     "sanitized_docReview_response", ""
        # )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a travel API integration documentation expoert."),
                (
                    "user",
                    "Write documentation for the following integration."
                    f"1. Backend endpoint: {sanitized_backend_endpoint_response}."
                    f"2. Frontend component: {sanitised_frontend_generated_code}."
                    f"3. API Provider docs: {request.docslink}"
                    "It should contain the following sections: Quick start guide, testing options (not that we have written tests at integration_tests.py), troubleshooting guide, support contact info, links to API provider docs.",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        context = {
            "input": "",
            "chat_history": request.chat_history,
            "docslink": request.docslink,
            "sanitised_frontend_generated_code": sanitised_frontend_generated_code,
            "sanitized_backend_endpoint_response": sanitized_backend_endpoint_response,
        }
        agent = create_openai_functions_agent(
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        documentation_result = response.get(
            "output", "No documentation action performed."
        )
        print(f"Documentation result: {documentation_result}")

        if capabilities_endPoints:
            endpoint_for_filename = capabilities_endPoints[0].replace("/", "_")
            documentation_file_name = (
                f"TechnicalDocumentation_{endpoint_for_filename}.txt"
            )
        else:
            documentation_file_name = "TechnicalDocumentation.txt"

        documentation_file_name = documentation_file_name.replace(":", "_").replace(
            "?", "_"
        )

        project_id = request.project
        doc_ref = db.collection("projectFiles").document(documentation_file_name)
        doc_ref.set(
            {
                "code": documentation_result,
                "createdAt": datetime.now(),
                "name": documentation_file_name,
                "project": db.collection("projects").document(project_id),
            }
        )
        print(f"Documentation saved as {documentation_file_name}")

        session_store[session_id] = {
            "step": 9,
        }

        formatted_documentation_result = format_response(documentation_result)
        return {
            "step": 8,
            "message": "Documentation sent",
            "output": formatted_documentation_result,
        }

    elif session_data["step"] == 9:
        print("Entering Step 9: API Key section...")

        prompt = ChatPromptTemplate.from_messages(
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
            llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0),
            tools=tools,
            prompt=prompt,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        response = await agent_executor.ainvoke(context)
        backend_apiKey_result = response.get(
            "output", "No backend endpoint action performed."
        )

        session_store[session_id] = {
            "step": 2,
        }

        steps_list = backend_apiKey_result.split("\n")

        return {
            "step": 9,
            "message": "API Key info sent",
            "output": steps_list,
        }


def format_response(frontend_function_response):
    cleaned_response = re.sub(r"```(python|jsx)\n?", "", frontend_function_response)
    cleaned_response = cleaned_response.replace("```", "")
    return cleaned_response


# Root Route
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Server
if __name__ == "__main__":
    uvicorn.run(
        "serve:app", host="localhost", port=8000, log_level="debug", reload=True
    )
