import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("Chatbot application using streamlit and OpenAI")

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]= "QnA Application using OpenAI"

system_message = SystemMessagePromptTemplate.from_template("You are an assistant with immense knowledge in world affairs.")
human_message = HumanMessagePromptTemplate.from_template("Question:{question}")

prompt = ChatPromptTemplate.from_messages(
    [system_message,human_message])

# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are an assistant with immense knowledge in world affairs."),
#     ("user","Question:{question}")
#     ])


# Sidebar
api_key = st.sidebar.text_input("Please enter the API key",type="password")
model = st.sidebar.selectbox("Please select the model.",["gpt-3.5-turbo","gpt-4o"])
temperature = st.sidebar.slider("Chose the temperature.",min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Chose maximum number of token.",min_value=50,max_value=300,value=150)


###############################################################################################################
####  Custom function that takes in a query and other model parameters and returns response from llm   ########
###############################################################################################################

def get_llm_response(question, model, api_key, temperature, max_token):
    llm = ChatOpenAI(model=model, openai_api_key = api_key, temperature=temperature)
    parser = StrOutputParser()

    lcel_Chain = prompt | llm | parser

    response = lcel_Chain.invoke({"question":{question}})
    return response


user_input = st.text_input("User:")
print(f"User query is: {user_input}")

if user_input and api_key:
    response_from_llm = get_llm_response(user_input, model, api_key, temperature, max_token)
    print(f"=================Response==================")
    print(response_from_llm)
    
    st.write("AI:",response_from_llm)

