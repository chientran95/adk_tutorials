from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

import yfinance as yf
from datetime import datetime


def get_stock_price(ticker: str) -> float:
    """
    Fetches the current stock price for the given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        float: The current stock price.
    """
    stock = yf.Ticker(ticker)
    price = stock.info.get('currentPrice', 'Price not available')
    return {'ticker': ticker, 'price': price}

def get_current_time() -> dict:
    """
    Returns the current system time as a string.

    Returns:
        dict: Status and result or error message.
    """
    now = datetime.now()
    report = f'The current time is {now.strftime("%H:%M:%S")}'

    return {'status': 'success', 'report': report}

time_agent = Agent(
    name='time_agent',
    description='A helpful assistant that provides the current system time.',
    instruction='You are a time assistant. Always use the get_current_time tool.',
    model=LiteLlm(model='ollama_chat/qwen2.5:7b'),
    tools=[get_current_time],
)

base_agent = LlmAgent(
    name='stock_price_agent',
    description='A helpful assistant that get stock price.',
    instruction=(
        'You are a stock price assistant. Always use the get_stock_price tool.'
        'Include the ticker symbol in your response.'
        'You have access to a specialist sub-agent called time_agent that can provide the current system time.'
    ),
    model=LiteLlm(model='ollama_chat/qwen2.5:7b'),
    # model=LiteLlm(
    #     api_base='http://localhost:11434/v1',
    #     model='openai/qwen2.5:7b',
    #     api_key='ollama', # Replace with your actual API key if needed
    # ),
    tools=[get_stock_price],
    sub_agents=[time_agent],
)

root_agent = base_agent
