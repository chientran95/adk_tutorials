This repository is to develop agents using Google agent development kit ([google-adk](https://google.github.io/adk-docs/)) sdk

## Samples
.
- `agent_ollama`: Tutorial with agent and sub-agent using qwen2.5 model in Ollama.
- `qdrant_rag`: Agent with RAG built using Qdrant as vector DB.
- `agent_team`: Agent collaboration tutorials.
  - `uv run agent_team/weather_agent.py`: Single agent and tool with simple session and runner
  - `uv run agent_team/weather_agent_team.py`: Agent team with simple session
  - `uv run agent_team/weather_agent_team_context.py`: Agent team with statefull session
  - `uv run agent_team/weather_agent_team_context.py --test_model_guardrail`: Agent team with statefull session and test for before LLM guardrail
  - `uv run agent_team/weather_agent_team_context.py --test_tool_guardrail`: Agent team with statefull session and test before tool guardrail
- `a2a_tutorial`:
  - Tutorial exposing agent to use A2A protocol
    1. `. ../.venv/bin/activate`
    2. `uvicorn a2a_root.remote_a2a.hello_world.agent:a2a_app --host localhost --port 8001`
    3. Access to check remote agent is up and running at `http://localhost:8001/.well-known/agent-card.json`
    4. In a separate terminal, run consuming agent with `adk web`

  - Tutorial consuming agent with A2A protocol
    1. `. ../.venv/bin/activate`
    2. `adk api_server --a2a --port 8001 a2a_basic/remote_a2a`
    3. Access to check remote agent is up and running at `http://localhost:8001/a2a/check_prime_agent/.well-known/agent-card.json`
    4. In a separate terminal, run consuming agent with `adk web`