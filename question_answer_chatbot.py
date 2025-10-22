import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries clearly and concisely."),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_response(question, api_key, model_name, temperature, max_tokens):
    os.environ["OPENAI_API_KEY"] = api_key
    llm_instance = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm_instance | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Streamlit UI
st.title("💬 Q&A Chatbot")

api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

model_name = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])

max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=200)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

st.write("Go ahead and ask your question below 👇")

user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar first.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please type your question above to start chatting.")
