import os
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI( model="gpt-5-mini",temperature=1,api_key=os.getenv("OPENAI_API_KEY"))

def booking_handler(request: str) -> str:
    print("\n Delegating to booking handler ---")
    return f"Booking handler received the request: '{request}'. Result: Simulated booking action"

def info_handler(request: str) -> str:
    print("\n Delegating to information handler ---")
    return f"Information handler received the request: '{request}'. Result: Simulated information action"


def unclear_handler(request: str) -> str:
    print("\n Delegating to unclear handler ---")
    return f"Unclear handler received the request: '{request}'. Result: Simulated No action"


coordinator_routing_prompt = ChatPromptTemplate.from_messages(
    [ ("system", """You are a coordinator that routes user requests to the appropriate handler based on the content of the request.
       The handlers available are: 'booking_handler', 'info_handler', and 'unclear_handler'.
         - If requests related to making reservations or appointments or bookings , give output as 'Booker'
         - If requests seeking information or details, output should be 'Information'
        - If requests are ambiguous or do not fit into the above categories, output should be 'Unclear'
        ONLY Return one word from above mentioned category.""")
        , ("user", "{request}")
    ]
)

coordinator_routing_chain = coordinator_routing_prompt | llm | StrOutputParser()

branches = { 
            "Booker": RunnablePassthrough.assign(output = lambda x: booking_handler(x["request"]["request"])),
            "Information": RunnablePassthrough.assign(output = lambda x: info_handler(x["request"]["request"])),
            "Unclear": RunnablePassthrough.assign(output = lambda x: unclear_handler(x["request"]["request"]))
            }


delegation_branch = RunnableBranch(
    
    (lambda x: x["category"] == "Booker", branches["Booker"]),
    (lambda x: x["category"] == "Information", branches["Information"]),
     branches["Unclear"]
)


coordinator_agent = ( { "request": RunnablePassthrough() ,
                        "category": coordinator_routing_chain
                        } 
                      | delegation_branch |  (lambda x: x["output"])
                     )
                       

result = coordinator_agent.invoke({"request": "Lets dance in the party tonight?"})
print(result)