import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it",api_key=my_api_key)

system_message = SystemMessagePromptTemplate.from_template("You are an expert on Geopgraphy")
human_message = HumanMessagePromptTemplate.from_template("Explain in 100 words about {topic}")

prompt = ChatPromptTemplate.from_messages(messages=[
   system_message,human_message
])

# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are an expert on Geopgraphy"),
#     ("human","Explain in 100 words about {topic}")
# ])

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"topic":"Indian influence in Hisory"})
print("\n========Response==========\n")
print(response)
