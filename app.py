from dotenv import load_dotenv
import os
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

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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

# Example chat history
chat_history = [
    HumanMessage(
        content="Can you help me integrate my codebase with Duffel Flights Search capability?"
    ),
    AIMessage(content="Yes!"),
]

# Invoke the retrieval chain with conversation history and input
response = retrieval_chain.invoke(
    {
        "chat_history": chat_history,
        "input": "What was the origin and destination of the first slice?",
    }
)

# Print the response
print(response["answer"])
