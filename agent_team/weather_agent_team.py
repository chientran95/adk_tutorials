import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from weather_agent import get_weather, call_agent_async
from reception_agents import greeting_agent, farewell_agent

print("Libraries imported.")


AGENT_MODEL = os.getenv("MODEL_GPT_4")

weather_agent_team = Agent(
    name="weather_agent_v2",
    model=LiteLlm(model=AGENT_MODEL),
    description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
    instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                "You have specialized sub-agents: "
                "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                "If it's a weather request, handle it yourself using 'get_weather'. "
                "For anything else, respond appropriately or state you cannot handle it.",
    tools=[get_weather],
    sub_agents=[greeting_agent, farewell_agent]
)
print(f"✅ Root Agent '{weather_agent_team.name}' created using model '{AGENT_MODEL}' with sub-agents: {[sa.name for sa in weather_agent_team.sub_agents]}")

async def run_team_conversation():
    print("\n--- Testing Agent Team Delegation ---")
    session_service = InMemorySessionService()
    APP_NAME = "weather_tutorial_agent_team"
    USER_ID = "user_1_agent_team"
    SESSION_ID = "session_001_agent_team"
    _ = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    runner_agent_team = Runner(
        agent=weather_agent_team,
        app_name=APP_NAME,
        session_service=session_service
    )
    print(f"Runner created for agent '{weather_agent_team.name}'.")

    await call_agent_async(query = "Hello there!",
                           runner=runner_agent_team, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async(query = "What is the weather in New York?",
                           runner=runner_agent_team, user_id=USER_ID, session_id=SESSION_ID)
    await call_agent_async(query = "Thanks, bye!",
                           runner=runner_agent_team, user_id=USER_ID, session_id=SESSION_ID)

if __name__ == "__main__":
    print("Executing using 'asyncio.run()' (for standard Python scripts)...")
    try:
        asyncio.run(run_team_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
