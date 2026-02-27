import os
from crewai import Agent, Task, Crew
from crewai.tools import tool
import logging
import asyncio
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage , SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable, RunnableParallel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s'
                    )


load_dotenv()

llm = ChatOpenAI( model="gpt-5-mini",temperature=1,api_key=os.getenv("OPENAI_API_KEY"))

@tool("stock price tool")
def get_stock_price(ticker: str) -> float:
    """
    

    Args:
        ticker (str): _description_

    Returns:
        float: _description_
    """
    logging.info(f"Fetching stock price for {ticker}")
    simulated_prices = {
        "AAPL": 148.25,
        "GOOGL": 2800.50,
        "AMZN": 3400.75,
        "MSFT": 299.99,
        "TSLA": 700.00
    }
    
    price = simulated_prices.get(ticker.upper(), None)
    if price is None:
        logging.warning(f"Ticker {ticker} not found in simulated prices.")
        return -1.0
    logging.info(f"Price for {ticker} is {price}")
    return price


# Defining agent 
    
financial_agent = Agent(
    role = 'Financial Advisor',
    goal = 'Provide financial advice based on stock prices and market trends',
    backstory = 'You are a seasoned financial advisor with expertise in stock market analysis and investment strategies. Your role is to assist clients in making informed financial decisions by providing insights based on stock prices and market trends.',
    tools = [get_stock_price]
)   
    
# Defining task

analyzing_stock_task = Task(
    description=( " what is the current price of the apple ( ticker : AAPL) stock ? " 
                 "Use the 'stock price tool' to find it "
                 "if ticket is not found , you must report that you were unable to retrieve the price")
    , 
    expected_output=(" A single , clear sentence statting the simulated stock proce for AAPS."
    "For example : 'The current price of AAPL stock is $150.25."
    "If the ticker is not found, respond with 'Unable to retrieve the price for AAPL.'")
, 
agent = financial_agent,
)

# Combine the pieces

financial_crew = Crew(
    agents=[financial_agent],
    tasks=[analyzing_stock_task],
    verbose=True
)

def main():
    print ("Starting the financial crew to analyze stock price...")
    result = financial_crew.kickoff()
    print("Crew Result: ", result)
    
if __name__ == "__main__":
    main()
    