import fitz  # PyMuPDF
import re
import numpy as np
import tiktoken
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import os
from dotenv import load_dotenv
load_dotenv()

# Load model and tokenizer once
embedder = SentenceTransformer('all-MiniLM-L6-v2')
encoding = tiktoken.get_encoding("cl100k_base")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

def generate_embeddings_with_keywords(
    data: List[Dict],
    model: str = "multilingual-e5-large",
    index_name: str = "userdocindeexx",
    dimension: int = 1024,
    region: str = "us-east-1",
    cloud: str = "aws",
    create_index: bool = True
) -> List[List[float]]:
    """
    Creates embeddings for a list of chunks with keyword weighting.

    Args:
        data (list): List of dicts with "chunk" and "keywords" fields.
        model (str): Pinecone-supported embedding model.
        index_name (str): Pinecone index name.
        dimension (int): Dimensionality of embeddings.
        region (str): Pinecone region.
        cloud (str): Pinecone cloud provider.
        create_index (bool): Whether to create the index if it doesn't exist.

    Returns:
        list: List of embeddings.
    """
    if create_index:
        print(f"Creating index '{index_name}' if it doesn't exist...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )

    # Prepare weighted text by repeating or prepending keywords
    contents = []
    for entry in data:
        chunk = entry.get("chunk", "")
        keywords = entry.get("keywords", [])
        if keywords:
            # Weight keywords: repeat or prepend them
            weighted_keywords = " ".join(keywords * 3)  # Repeat 3x
            full_text = f"{weighted_keywords}. {chunk}"
        else:
            full_text = chunk
        contents.append(full_text)

    print("Sending data for embedding...")
    embeddings = pc.inference.embed(
        model=model,
        inputs=contents,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    print("✅ Embedding complete. Sample vector:")
    # print(embeddings[0])
    
    # Ensure Pinecone index is loaded
    index = pc.Index(index_name)
    print("✅ Vectors upserting...")

    vectors = []
    for entry, embedding in zip(data, embeddings):
        # Use a meaningful or unique ID, fallback to entry ID
        vector_id = str(entry.get('id', 'unknown'))

        vectors.append({
            "id": vector_id,
            "values": embedding['values'],  # or just `embedding` if it's a list
            "metadata": {
                "chunk": entry.get('chunk', ''),
                "Sub Title": entry.get('Sub Title', ''),
                "keywords": entry.get('keywords', [])
            }
        })

    # Upsert into the Pinecone index
    index.upsert(
        vectors=vectors,
        namespace="ns"  # Replace with your actual namespace if needed
    )

    # Optional: Check index stats
    print("✅ Vectors upserted. Printing describe_index_stats():")
    return embeddings