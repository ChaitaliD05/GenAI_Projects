import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


#prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Your the helpful assistant plz answer the questions asked."),
        ("user","Question:{question}")
         
    ]
)

st.title("Ollama - gemma2b application")

# llm
llm = Ollama(model="gemma:2b")

#parser
output_parser = StrOutputParser()

#chain
chain= prompt |llm|output_parser


input_text= st.text_input("What question do u have?")
if input_text:
    st.write(chain.invoke({"question":input_text}))

