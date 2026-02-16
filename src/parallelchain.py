import os
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI( model="gpt-5-mini",temperature=1,api_key=os.getenv("OPENAI_API_KEY"))


summarize_chain : Runnable = (
ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that summarizes user requests into concisely: "),
        ("user", "{topic}")
    ]
)
| llm | StrOutputParser()
)

question_chain : Runnable = ( 
ChatPromptTemplate.from_messages(
    [
        ("system", " Generate the three interesting questions about the following topic: "),
        ("user", "{topic}")
    ]
)
| llm | StrOutputParser()
)

terms_chain : Runnable = (
ChatPromptTemplate.from_messages(
    [
        ("system", "Identify 5-10 key terms and concepts related to the following topic: "),
        ("user", "{topic}")
    ])
| llm | StrOutputParser()
)   

