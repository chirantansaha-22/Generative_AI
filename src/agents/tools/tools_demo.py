
##########################################################################################################
### Tools are a way to encapsulate a function and its schema in a way that can be passed to a chat model.
### The tool can be called by the chat model with specific inputs and the tool will execute the function.
###    
### This allows the chat model to request the execution of a specific function with specific inputs.
### The tool will execute the function and return the output to the chat model.
############################################################################################################

from langchain_core.tools import tool

"""
LangChain supports the creation of tools from:

1) Functions
2) LangChain Runnables
3) By sub-classing from BaseTool 

"""
###################################################
### 1) Creating a tool from a function
###################################################
@tool
def multiply(a:int, b:int):
    """
    Multiply two numbers.
    """
    return a * b

product = multiply.invoke({"a":2,"b":5})
print("Printing tool parameters\n============================")
print(f"Tool name:{multiply.name} ")
print(f"Tool arguments:{multiply.args} ")
print(f"Tool description:{multiply.description} ")

print(f"Calling/using the tool\n============================")
print(f"product: {product}")

#################################################
#########       Using Annotated     #############
#################################################
from typing import Annotated, List

@tool
def multiply_by_max(
       a: Annotated[int,"First integer"],
       b: Annotated[List[int],"list of integers to choose max from"],
       c: Annotated[int,"3rd integer"] 
) -> int:
    "multipy a with max(b) and then multiplied by c"
    return a * max(b) * c

a = 2
b=[3,5,6,8]
c=10
print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")
print(f"Result: {multiply_by_max.invoke({'a':a,'b':b,'c':c})}")

#################################################
#########       Using Field     #################
#################################################
from pydantic import Field, BaseModel



class MultiplyByMaxInput(BaseModel):
    a: int = Field(..., description="First integer")
    b: List[int] = Field(..., description="list of integers to choose max from")
    c: int = Field(..., description="3rd integer")



@tool("function1", args_schema=MultiplyByMaxInput, return_direct=True)
def multiply_by_max(a:int, b:List[int], c:int) ->int:
    "multipy a with max(b) and then multiplied by c"
    return a * max(b) * c

a = 3
b=[3,5,6,8]
c=30

output = multiply_by_max.invoke({'a':a,'b':b,'c':c})  
print(f"Result: {output}")