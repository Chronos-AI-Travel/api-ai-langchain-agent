from dotenv import load_dotenv
import os
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize embeddings and load documents
embeddings = OpenAIEmbeddings()
loader = WebBaseLoader("https://duffel.com/docs/guides/getting-started-with-flights")
docs = loader.load()

# Split documents and create a vector store
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Initialize the LLM and output parser
llm = ChatOpenAI(openai_api_key=openai_api_key)
output_parser = StrOutputParser()

# Setup the retriever
retriever = vector.as_retriever()

# Define the prompt for generating search queries based on conversation history
search_query_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ),
    ]
)

# Create a history-aware retriever
retriever_chain = create_history_aware_retriever(llm, retriever, search_query_prompt)

# Setup retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Initiate Tavily search tool with the API key
search = TavilySearchResults(tavily_api_key=tavily_api_key)

# List tools
tools = [retriever_tool, search]

# Define the prompt for the document chain
document_prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

# Create the document chain
document_chain = create_stuff_documents_chain(llm, document_prompt)

# Combine into a retrieval chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# New code to integrate
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example chat history
chat_history = [
    HumanMessage(
        content="Can you help me integrate my codebase with Duffel Flights Search capability?"
    ),
    AIMessage(content="Yes!"),
]

# Now, invoke the agent_executor with a new question about LangSmith
response = agent_executor.invoke({"input": "Give me code snippets that I can paste in to make Duffel Flights Search work"})

# Print the response
print(response)