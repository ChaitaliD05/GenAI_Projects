# pip install langchain langgraph langsmith requests python-dotenv

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- Set LangSmith Tracking (for tracing) ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "StockPriceDemo"

# --- Initialize LLM ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Define Graph State ---
class StockState(dict):
    query: str
    stock_symbol: str
    stock_price: str
    final_answer: str


# --- 1️⃣ Extract Stock Symbol ---
def extract_symbol(state: StockState):
    """Ask LLM to identify the stock ticker from user query"""
    prompt = f"Extract only the stock symbol (e.g., AAPL, TSLA) from: {state['query']}"
    result = llm.invoke(prompt)
    state["stock_symbol"] = result.content.strip().upper()
    return state


# --- 2️⃣ Fetch Stock Price ---
def fetch_price(state: StockState):
    """Fetch stock price using Yahoo Finance API"""
    symbol = state["stock_symbol"]
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        data = requests.get(url).json()
        price = data["quoteResponse"]["result"][0]["regularMarketPrice"]
        state["stock_price"] = f"{symbol}: ${price:.2f}"
    except Exception:
        state["stock_price"] = f"Could not fetch price for {symbol}"
    return state


# --- 3️⃣ Generate Final Response ---
def generate_response(state: StockState):
    """Use LLM to summarize the result for user"""
    prompt = f"User asked: {state['query']}\nResult: {state['stock_price']}\nWrite a one-line answer."
    result = llm.invoke(prompt)
    state["final_answer"] = result.content.strip()
    return state


# --- Build LangGraph Workflow ---
graph = StateGraph(StockState)
graph.add_node("extract_symbol", extract_symbol)
graph.add_node("fetch_price", fetch_price)
graph.add_node("generate_response", generate_response)
graph.set_entry_point("extract_symbol")
graph.add_edge("extract_symbol", "fetch_price")
graph.add_edge("fetch_price", "generate_response")
graph.add_edge("generate_response", END)


app = graph.compile()

# --- Run the Tool ---
if __name__ == "__main__":
    query = input("Ask for a stock price (e.g., 'What is the price of AAPL today?'): ")
    result = app.invoke({"query": query})
    print("\n✅ Final Answer:")
    print(result["final_answer"])
