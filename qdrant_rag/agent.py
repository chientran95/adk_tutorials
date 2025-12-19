import os
import uuid
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv


load_dotenv()

required_vars = ["QDRANT_URL", "QDRANT_COLLECTION_NAME", "QDRANT_VECTOR_NAME"]

if all(var in os.environ for var in required_vars):
    print("All required environment variables are set")
else:
    missing = [var for var in required_vars if var not in os.environ]
    print("Missing env vars:", missing)

def qdrant_setup():
    """Setup Qdrant client and model for embeddings."""


    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    vector_name = os.getenv("QDRANT_VECTOR_NAME")

    client = QdrantClient(url=qdrant_url)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    return client, model, collection_name, vector_name

client, model, collection_name, vector_name = qdrant_setup()

def qdrant_find(query: str) -> dict:
    """
    Perform a vector search over the Qdrant collection.

    Args:
        query (str): The query string to search for.

    Returns:
        dict: The search results from the Qdrant collection.
    """

    test_embedding = model.encode([query])[0]
    hits = client.query_points(
        collection_name=collection_name,
        query=test_embedding,
        using=vector_name,
        limit=5,
    )
    query_results = ""
    for result in hits.points:
        query_results += result.payload['origin_text'] + "\n" + ("-"*20) + "\n"

    return {"results": query_results.strip()}

def qdrant_add(query: str) -> str:
    """
    Add new document to the Qdrant collection.

    Args:
        query (str): The document text to add.

    Returns:
        str: Confirmation message.
    """

    try:
        embeddings = model.encode(query)

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        vector_name: embeddings
                    },
                    payload={"origin_text": query}
                )
            ]
        )
        return "Document added successfully."
    except Exception as e:
        return f"Error adding document: {str(e)}"

root_agent = Agent(
    model=LiteLlm(model='ollama_chat/qwen2.5:7b'),
    name="qdrant_agent",
    instruction=(
        "Help users store and retrieve information using semantic search. "
        "You have access to a Qdrant tool that allows you to perform vector searches "
        "over a collection of documents named qdrant_find "
        "and a tool to add new documents to the collection named qdrant_add. "
    ),
    tools=[
       qdrant_find, qdrant_add
    ],
)
