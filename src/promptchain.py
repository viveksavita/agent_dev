import os
from langchain_openai import ChatOpenAI , OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI( model="gpt-5-mini",temperature=1,api_key=os.getenv("OPENAI_API_KEY"))

# Prompt 1 : extract information from a text

prompt_extract = ChatPromptTemplate.from_template( 
"Extract technical information from the following text :\n\n{text_input} "
)

# Prompt 2 : Transform the extracted information into a JSON format

prompt_json = ChatPromptTemplate.from_template(
"Transform the following technical information into a JSON format with 'cpu' , 'memory' , 'storage' as keys:\n\n{extracted_info}"
)

# Building the chain using LCEL

extraction_chain = prompt_extract | llm | StrOutputParser()

full_chain = ( {"extracted_info": extraction_chain} 
              | prompt_json | llm | StrOutputParser()   )

input_text = "The server has 16GB of RAM, a 2.4GHz quad-core CPU, and 1TB of storage."

result = full_chain.invoke({"text_input": input_text})

print(result)