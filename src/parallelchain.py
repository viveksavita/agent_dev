import os
import asyncio
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable, RunnableParallel
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

map_chain = RunnableParallel( 
                             {
        "summary": summarize_chain,
        "questions": question_chain,
        "terms": terms_chain,
        "topic": RunnablePassthrough(),
    }
    )

sysnthesis_prompt = ChatPromptTemplate.from_messages(
    [
        ( "system", """ Based on the following information :
         Summary: {summary}
         Questions: {questions}
         Terms: {terms}
         Sysnthesize a compreshensive answer."""),
        ("user", "{topic}" )
    ]
)

full_chain = map_chain | sysnthesis_prompt | llm | StrOutputParser()


async def parallel_chain(topic: str) -> None:
    result = await full_chain.ainvoke(topic)
    print("\n Final Synthesis Result: \n", result)
    
    
if __name__ == "__main__":
    test_topc = "The impact of climate change on global agriculture"
    asyncio.run(parallel_chain(test_topc))