import os
from dotenv import load_dotenv
#from langchain_community.llms import Ollama #will be deprecated in v1.0.0
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

llm = OllamaLLM(model="gemma2:2b")

#############################################################
######################                  #####################
#####################     Direct usage     ##################
######################                  #####################
#############################################################


# response = llm.invoke({"BP"})
response = llm.invoke("please tell about BJP party's election strategy in 100 words")

print(response)

#############################################################
######################                  #####################
#####################     Using LCEL     ####################
######################                  #####################
#############################################################

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an expert on politics"),
    ("human","please tell about {party} party's election strategy in 100 words")
])

print("\nResponse using LCEL chain\n------------------------\n")

chain = prompt | llm
response_with_lcel = chain.invoke({"party":"BJP"})
print(response_with_lcel)

#print(os.getenv("OPENAI_API_KEY"))

