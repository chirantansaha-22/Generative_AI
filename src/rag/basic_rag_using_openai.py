import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Please respond to the question asked"), 
#     ("user", "Question:{question}")])

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only and provide accurate respone based on the question
    
    {context}
   
    Question:{input}

    """

)


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
openai_embedding = OpenAIEmbeddings()
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# For entire directory
#loader = PyPDFDirectoryLoader(directory_path="./data")

# For single file
loader = PyPDFLoader(file_path="./data/LLM.pdf")
docs = loader.load_and_split(splitter)
# print(docs[0])
# print(len(docs)) # 318


# For performance, only using first 50 documents
vector_store = FAISS.from_documents(documents=docs[:50], embedding=openai_embedding)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

question = input("Enter the query to search in the document: ")
response = retrieval_chain.invoke({"input": question})

print("Response from RAG\n========================")
print(response['answer'])


