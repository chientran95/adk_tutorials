import os
import argparse
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

required_vars = ["QDRANT_COLLECTION_NAME", "QDRANT_VECTOR_NAME"]

if all(var in os.environ for var in required_vars):
    print("All required environment variables are set")
else:
    missing = [var for var in required_vars if var not in os.environ]
    print("Missing env vars:", missing)

argparser = argparse.ArgumentParser(description="Push data to Qdrant collection")
argparser.add_argument('--data-path', type=str, required=False, help='Path to the data file')
args = argparser.parse_args()

df = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
passage_list = df.iloc[:500]['passage'].to_list()

if args.data_path:
    client = QdrantClient(path=args.data_path)
else:
    client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config={
        os.getenv("QDRANT_VECTOR_NAME"): VectorParams(size=384, distance=Distance.COSINE),
    }
)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(passage_list)

print(f"Pushing {len(embeddings)} vectors to Qdrant collection '{os.getenv("QDRANT_COLLECTION_NAME")}'...")
client.upsert(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    points=[
        PointStruct(
            id=idx,
            vector={
                os.getenv("QDRANT_VECTOR_NAME"): vector.tolist()
            },
            payload={"origin_text": passage_list[idx]}
        )
        for idx, vector in enumerate(embeddings)
    ]
)



### Test retrieval
query_sentence = passage_list[3].split('.')[0]  # Use the first sentence of a sample data as a query
test_embedding = model.encode([query_sentence])[0]
hits = client.query_points(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    query=test_embedding,
    using=os.getenv("QDRANT_VECTOR_NAME"),
    limit=5,
)
for result in hits.points:
    print(result.payload['origin_text'])
    print("-"*20)
