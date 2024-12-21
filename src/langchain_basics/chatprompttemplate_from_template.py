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
######## 1) Using from_template(string) 
########    Create a ChatPromptTemplate instance from a single template string
########    Use case: straightforward templates using direct prompts.
##############################################################################

prompt_using_template = ChatPromptTemplate.from_template("Explain the importance of {topic} in mathematics in less than 100 words")

# Creating a LCEL chain of runnables
lcel_chain = prompt_using_template | llm

response = lcel_chain.invoke({"topic":"Derivatives"})
print(f"Response from LLM by using prompt_using_template")
print("==================================================")
print(response.content)






template = ChatPromptTemplate.from_template("Please explain the future of {topic} in less than 50 words.")
prompt = template.format_prompt(topic="AI and ML")

list_of_messages = prompt.to_messages()

# Convert the formatted prompt to messages
response_without_lcel = llm.invoke(list_of_messages)

print("Response without using LCEL")
print("==============================")
print(response_without_lcel.content)

#chain = prompt | llm

#response = chain.invoke()