from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv


llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"),streaming=True)


###########################################################################################
#### Notes:
#### Langgraph is a library for building stateful, multi actor applications with LLMs,
#### used to create agent and multi-agent workflows
###########################################################################################

#######################################################################################################
######################################### 1. Define the State for the Graph ###########################
#######################################################################################################
## The state contains the graph's schema and teh reducer functions
# State is a TypedDict with only one key: "messages"
# add_messages used below is a reducer function that append new messages to the list instead of overwriting it.
class State(TypedDict):
    messages: Annotated[list, add_messages] 



#######################################################################################################
#################################### 2. Define the nodes for the graph_builder #############################
#######################################################################################################

## Important points:
## Nodes represent units of work and are generally python functions. 
## Each node can receive the current "State" and output the "State" with some updates



def chatbot(state: State) -> State:
    return {"messages": llm.invoke(state["messages"])}

#######################################################################################################
#################################### 3. Initialize the Graph ##########################################
#################################### 4. Add node to the Graph #########################################
#################################### 5. Add edges to the Graph ########################################
#######################################################################################################
graph_builder = StateGraph(State)


""" Adding a 'chatbot' node"""
graph_builder.add_node("chatbot", chatbot) 

"""Adding an entry point"""
graph_builder.add_edge(START, "chatbot")

"""Adding an exit  point"""
graph_builder.add_edge("chatbot", END)

###################################################################################################################
############## 6. Compile the graph builder with compile() to get a "CompiledGraph", which can be invoked   #######
###################################################################################################################
graph = graph_builder.compile()


########################################################################
##################  7. Display the graph(optional) #####################
########################################################################
from IPython.display import Image, display


# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass

from colorama import Fore, Style

#print(graph.invoke({"messages":[("user",user_input)]}))


while True:
    user_input = input(Fore.RED + "What's your query?\n" + Style.RESET_ALL)
    if user_input.lower() in ["quit","exit","q","e"]:
        print(Fore.GREEN + "Good Bye! " + Style.RESET_ALL)
        break
    
    for event in graph.stream({"messages":[("user",user_input)]}):
        for value in event.values():
            print(Fore.LIGHTGREEN_EX + "Assistant: "+ Style.RESET_ALL,value["messages"].content)




