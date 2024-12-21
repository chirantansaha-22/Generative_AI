import os

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

my_openai_key = os.getenv("OPENAI_API_KEY")




######################################################################
######################                             ###################
######################    Instantiation Methods    ###################
######################   ========================  ###################
######################    1) from_template()       ###################
######################    2) from_messages()       ###################
######################    3) from_strings()        ###################
######################    4) from_role_strings()   ###################
######################    4) from_orm()            ###################
######################    4) from json/yaml        ###################
######################                             ###################
######################################################################

# Define the model
# we will be using the same model for all the various methods of instatntiating the Prompt template

llm = ChatOpenAI(api_key=my_openai_key, model="gpt-3.5-turbo")

##############################################################################
######## 2) Using from_messages(string) 
########    The from_messages() method is extremely flexible and accepts various 
########    representations of messages, enabling you to mix and match different 
########    formats for defining chat prompts
########
########    Usages: a) Message type
########            b) 
##############################################################################
prompt_using_messages = ChatPromptTemplate.from_messages([
    ("system","You are an expert on Technology"),
    ("human", "What is the future of {sme} jobs in terms of Number of jobs?")
])


chain = prompt_using_messages | llm

response = chain.invoke({"sme":"Coding"})
print(response.content)