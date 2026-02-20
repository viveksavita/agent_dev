import os
import asyncio
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage , SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI( model="gpt-5-mini",temperature=1,api_key=os.getenv("OPENAI_API_KEY"))

def run_relflection_loop():
    """Demonstarte a multi-step AI reflection loop to progressively imporve a python function that calculates the factorial of a number."""
    
    task_prompt = """
    your task is to create a python function names `calculate_factorial`
    
    This function should do the following:
    - Take a single integer input `n`
    - Return the factorial of `n`
    - Include doc string explaining the function's purpose and usage
    - Handle edge cases such as negative numbers and non-integer inputs by returning an appropriate error message.
    """
    max_iterations = 3
    current_code = ""
    
    message_history = [HumanMessage(content=task_prompt)]
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        if i==0:
            print ("\n Stage 1: Initial Code Generation")
            response = llm.invoke(message_history)
            current_code = response.content
            
        else:
            print ("\n Stage 1: Self-Reflection and Improvement")
            message_history.append(HumanMessage(content="Please refine the code using the following feedback and only return code. Do not ask any preference and choice: "))
            response = llm.invoke(message_history)
            current_code= response.content
        
        message_history.append(response)
        
        print( "\n Stage 2 : Refelcting the generated code")
        
        refelctor_prompt = [
            SystemMessage( content="""You are a senior software engineer tasked with reviewing and improving the following python code.Critically evaluate the python code based on the original 
                          requirement. Look for the bugs , style issues, edge cases and overall code quality. Provide specific feedback for improvement in bullted list. 
                          Do not ask any suggestion for actions, just give feedback. 
                          If the code is perfect and meets most of the requirements respond with the single phrase 'CODE_IS_PERFECT' """),
            HumanMessage(content=f"""Here is the {task_prompt} \n\n Current Code: \n {current_code} \n\n )
                         """)
        ]
              
        critique_response = llm.invoke(refelctor_prompt)
        critique = critique_response.content
        
        print ("\n Stage 3: Critique and Feedback")
        print(critique)
        
        if "CODE_IS_PERFECT" in critique:
            print("\n The code is perfect and meets all the requirements. Ending the reflection loop.")
            print ("\n Final Code: \n", current_code)
            break
        
        message_history.append(HumanMessage(content=f"Here is the critique of the code: \n {critique}"))
        
        print("\n final refined code after reflection: \n", current_code)
        
        
if __name__ == "__main__":
    run_relflection_loop()