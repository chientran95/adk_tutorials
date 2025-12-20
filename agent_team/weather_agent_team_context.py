import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
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



def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather, converts temp unit based on session state."""
    print(f"--- Tool: get_weather_stateful called for {city} ---")

    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius") # Default to Celsius
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
            temp_unit = "°F"
        else: # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}

print("✅ State-aware 'get_weather_stateful' tool defined.")

AGENT_MODEL = os.getenv("MODEL_NAME")

weather_agent_team = Agent(
    name="weather_agent_v4_stateful",
    model=LiteLlm(model=AGENT_MODEL),
    description="Main agent: Provides weather (state-aware unit), delegates greetings/farewells, saves report to state.",
    instruction="You are the main Weather Agent. Your job is to provide weather using 'get_weather_stateful'. "
                "The tool will format the temperature based on user preference stored in state. "
                "Delegate simple greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
                "Handle only weather requests, greetings, and farewells.",
    tools=[get_weather_stateful], # Use the state-aware tool
    sub_agents=[greeting_agent, farewell_agent], # Include sub-agents
    output_key="last_weather_report"
)
print(f"✅ Root Agent '{weather_agent_team.name}' created using model '{AGENT_MODEL}' with sub-agents: {[sa.name for sa in weather_agent_team.sub_agents]}")

async def run_team_conversation():
    print("\n--- Testing Agent Team Delegation with context ---")
    session_service_stateful = InMemorySessionService()
    print("✅ New InMemorySessionService created for state demonstration.")

    # Define a NEW session ID for this part of the tutorial
    APP_NAME = "weather_tutorial_agent_team"
    USER_ID_STATEFUL = "user_state_demo"
    SESSION_ID_STATEFUL = "session_state_demo_001"

    # Define initial state data - user prefers Celsius initially
    initial_state = {
        "user_preference_temperature_unit": "Celsius"
    }

    # Create the session, providing the initial state
    _ = await session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID_STATEFUL,
        session_id=SESSION_ID_STATEFUL,
        state=initial_state # <<< Initialize state during creation
    )
    print(f"✅ Session '{SESSION_ID_STATEFUL}' created for user '{USER_ID_STATEFUL}'.")

    # Verify the initial state was set correctly
    retrieved_session = await session_service_stateful.get_session(app_name=APP_NAME,
                                                            user_id=USER_ID_STATEFUL,
                                                            session_id = SESSION_ID_STATEFUL)
    print("\n--- Initial Session State ---")
    if retrieved_session:
        print(retrieved_session.state)
    else:
        print("Error: Could not retrieve session.")

    runner_agent_team = Runner(
        agent=weather_agent_team,
        app_name=APP_NAME,
        session_service=session_service_stateful
    )
    print(f"Runner created for agent '{weather_agent_team.name}'.")
    print("\n\n\n--- Testing State: Temp Unit Conversion & output_key ---")

    # 1. Check weather (Uses initial state: Celsius)
    print("--- Turn 1: Requesting weather in London (expect Celsius) ---")
    await call_agent_async(query= "What's the weather in London?",
                           runner=runner_agent_team, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    
    # 2. Manually update state preference to Fahrenheit - DIRECTLY MODIFY STORAGE
    print("\n--- Manually Updating State: Setting unit to Fahrenheit ---")
    try:
        # Access the internal storage directly - THIS IS SPECIFIC TO InMemorySessionService for testing
        # NOTE: In production with persistent services (Database, VertexAI), you would
        # typically update state via agent actions or specific service APIs if available,
        # not by direct manipulation of internal storage.
        stored_session = session_service_stateful.sessions[APP_NAME][USER_ID_STATEFUL][SESSION_ID_STATEFUL]
        stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"
        print(f"--- Stored session state updated. Current 'user_preference_temperature_unit': {stored_session.state.get('user_preference_temperature_unit', 'Not Set')} ---") # Added .get for safety
    except KeyError:
        print(f"--- Error: Could not retrieve session '{SESSION_ID_STATEFUL}' from internal storage for user '{USER_ID_STATEFUL}' in app '{APP_NAME}' to update state. Check IDs and if session was created. ---")
    except Exception as e:
            print(f"--- Error updating internal session state: {e} ---")

    # 3. Check weather again (Tool should now use Fahrenheit)
    # This will also update 'last_weather_report' via output_key
    print("\n--- Turn 2: Requesting weather in New York (expect Fahrenheit) ---")
    await call_agent_async(query= "Tell me the weather in New York.",
                           runner=runner_agent_team, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    
    # 4. Test basic delegation (should still work)
    # This will update 'last_weather_report' again, overwriting the NY weather report
    print("\n--- Turn 3: Sending a greeting ---")
    await call_agent_async(query= "Hi!",
                           runner=runner_agent_team, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)
    
    print("\n\n\n--- Inspecting Final Session State ---")
    final_session = await session_service_stateful.get_session(app_name=APP_NAME,
                                                               user_id= USER_ID_STATEFUL,
                                                               session_id=SESSION_ID_STATEFUL)
    if final_session:
        # Use .get() for safer access to potentially missing keys
        print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
        print(f"Final Last Weather Report (from output_key): {final_session.state.get('last_weather_report', 'Not Set')}")
        print(f"Final Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful', 'Not Set')}")
        # Print full state for detailed view
        # print(f"Full State Dict: {final_session.state}") # For detailed view
    else:
        print("\n❌ Error: Could not retrieve final session state.")

if __name__ == "__main__":
    print("Executing using 'asyncio.run()' (for standard Python scripts)...")
    try:
        asyncio.run(run_team_conversation())
    except Exception as e:
        print(f"An error occurred: {e}")
