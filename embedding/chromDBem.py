from pinecone import Pinecone, ServerlessSpec
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)


# Load JSON data from a file
with open('dataKnow.json', 'r') as file:
    data = json.load(file)
# Extract content for embedding
contents = [entry['Content'] for entry in data if 'Content' in entry]
# print("content: ", contents)  
index_name = "refdocanalysis"

pc.create_index(
    name=index_name,
    dimension=1024, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=contents,
    parameters={"input_type": "passage", "truncate": "END"}
)
print("printing embeddings")
print(embeddings[0])



index = pc.Index(index_name)

vectors = []
for entry, embedding in zip(data, embeddings):
    if 'Content' in entry:
        vectors.append({
            "id": entry.get('Title', 'unknown_id'),  # Use Title as ID
            "values": embedding['values'],
            "metadata": {
                'Title': entry.get('Title', ''),
                'Sub Title': entry.get('Sub Title', ''),
                'Context': entry.get('Context', ''),
                'Content': entry.get('Content', [])
            }
        })
        


index.upsert(
    vectors=vectors,
    namespace="ns"
)# Wait for the index to be ready
print("printing describe_index_stats")
print(index.describe_index_stats())
