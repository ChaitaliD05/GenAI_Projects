from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
from langserve import add_routes

groq_api_key = os.getenv("GROQ_API_KEY")
model=ChatGroq(model="penai/gpt-oss-20b", groq_api_key=groq_api_key)


system_template="translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system',system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

chain=prompt_template|model|parser

app=FastAPI(title='langchain server',
            version="1.0",
            description="a simple api")

add_routes(
    app,
    chain,
    path="/chain"
)
if __name__== "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1", port = 8001)