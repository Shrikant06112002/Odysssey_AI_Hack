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

CHECKPOINT_KEYWORDS = [
    "Legal eligibility", 
    "State registration", 
    "Certifications", "Past performance", 
    "Disqualifying terms", 
    "Non-compliance alerts", 
    "Qualification standards",
    "Minimum experience", 
    "Industry certifications", 
    "Licenses", 
    "Background checks", 
    "Disqualifications", 
    "Credential verification",
    "Page limit", 
    "Font type/size", 
    "Line spacing", 
    "Table of contents", 
    "Required attachments", 
    "Submission cut-off", 
    "Delivery guidelines",
    "Unilateral termination rights", 
    "Exclusivity demands", 
    "Liability concerns", 
    "Penalties", 
    "Pricing constraints", 
    "Refund policies", 
    "Budget limitations"
]

def extract_keywords(text: str) -> list[str]:
    found_keywords = []
    lowered = text.lower()
    for keyword in CHECKPOINT_KEYWORDS:
        if keyword.lower() in lowered:
            found_keywords.append(keyword)
    return found_keywords

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def sentence_tokenize(text: str) -> list[str]:
    # Simple regex-based sentence splitter
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def embed_sentences(sentences: list[str]):
    return embedder.encode(sentences, convert_to_tensor=True)

def cluster_sentences(embeddings, threshold: float = 1.5):
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    clustering_model.fit(embeddings)
    return clustering_model.labels_

def group_by_clusters(sentences: list[str], labels: list[int]) -> list[list[str]]:
    clustered = {}
    for label, sentence in zip(labels, sentences):
        clustered.setdefault(label, []).append(sentence)
    return list(clustered.values())

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def merge_chunks(clusters: list[list[str]], max_tokens: int = 1500,overlap_tokens: int = 50) -> list[str]:
    final_chunks = []

    for cluster in clusters:
        current_chunk = []
        current_token_count = 0

        for sentence in cluster:
            sentence_token_count = count_tokens(sentence)

            # If adding sentence exceeds max_tokens, save current chunk and start new one with overlap
            if current_token_count + sentence_token_count > max_tokens:
                chunk_text = " ".join(current_chunk).strip()
                final_chunks.append(chunk_text)

                # Create overlap from last N tokens
                all_tokens = encoding.encode(chunk_text)
                overlap_token_ids = all_tokens[-overlap_tokens:]
                overlap_text = encoding.decode(overlap_token_ids)

                current_chunk = [overlap_text.strip(), sentence]
                current_token_count = count_tokens(current_chunk[0]) + sentence_token_count
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count

        # Add final leftover chunk
        if current_chunk:
            final_chunks.append(" ".join(current_chunk).strip())

    return final_chunks


def semantic_chunk_pdf_json(pdf_path: str, max_tokens: int = 1000, threshold: float = 1.5) -> list[dict]:
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("Tokenizing into sentences...")
    sentences = sentence_tokenize(raw_text)

    print("Generating embeddings...")
    embeddings = embed_sentences(sentences)
    embeddings_np = embeddings.cpu().numpy()

    print("Clustering sentences by semantics...")
    labels = cluster_sentences(embeddings_np, threshold=threshold)

    print("Grouping sentences into clusters...")
    sentence_clusters = group_by_clusters(sentences, labels)

    print("Merging clusters into token-limited chunks...")
    chunks = merge_chunks(sentence_clusters, max_tokens=max_tokens)

    print("Extracting keywords and formatting output...")
    json_output = []
    for idx, chunk in enumerate(chunks, start=1):
        keywords = extract_keywords(chunk)
        json_output.append({
            "id": idx,
            "chunk": chunk,
            "keywords": keywords
        })

    print("Creating embdding for user pdf")
    # embeddings = generate_embeddings_with_keywords(json_output)
    return json_output

# def generate_embeddings_with_keywords(
#     data: List[Dict],
#     model: str = "multilingual-e5-large",
#     index_name: str = "userdocindexx",
#     dimension: int = 1024,
#     region: str = "us-east-1",
#     cloud: str = "aws",
#     create_index: bool = True
# ) -> List[List[float]]:
#     """
#     Creates embeddings for a list of chunks with keyword weighting.

#     Args:
#         data (list): List of dicts with "chunk" and "keywords" fields.
#         model (str): Pinecone-supported embedding model.
#         index_name (str): Pinecone index name.
#         dimension (int): Dimensionality of embeddings.
#         region (str): Pinecone region.
#         cloud (str): Pinecone cloud provider.
#         create_index (bool): Whether to create the index if it doesn't exist.

#     Returns:
#         list: List of embeddings.
#     """
#     if create_index:
#         print(f"Creating index '{index_name}' if it doesn't exist...")
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(cloud=cloud, region=region)
#         )

#     # Prepare weighted text by repeating or prepending keywords
#     contents = []
#     for entry in data:
#         chunk = entry.get("chunk", "")
#         keywords = entry.get("keywords", [])
#         if keywords:
#             # Weight keywords: repeat or prepend them
#             weighted_keywords = " ".join(keywords * 3)  # Repeat 3x
#             full_text = f"{weighted_keywords}. {chunk}"
#         else:
#             full_text = chunk
#         contents.append(full_text)

#     print("Sending data for embedding...")
#     embeddings = pc.inference.embed(
#         model=model,
#         inputs=contents,
#         parameters={"input_type": "passage", "truncate": "END"}
#     )

#     print("✅ Embedding complete. Sample vector:")
#     # print(embeddings[0])
    
#     # Ensure Pinecone index is loaded
#     index = pc.Index(index_name)
#     print("✅ Vectors upserting...")

#     vectors = []
#     for entry, embedding in zip(data, embeddings):
#         # Use a meaningful or unique ID, fallback to entry ID
#         vector_id = str(entry.get('id', 'unknown'))

#         vectors.append({
#             "id": vector_id,
#             "values": embedding['values'],  # or just `embedding` if it's a list
#             "metadata": {
#                 "chunk": entry.get('chunk', ''),
#                 "Sub Title": entry.get('Sub Title', ''),
#                 "keywords": entry.get('keywords', [])
#             }
#         })

#     # Upsert into the Pinecone index
#     index.upsert(
#         vectors=vectors,
#         namespace="ns"  # Replace with your actual namespace if needed
#     )

#     # Optional: Check index stats
#     print("✅ Vectors upserted. Printing describe_index_stats():")


#     return embeddings


    # for i, chunk in enumerate(chunks):
    #     print(f"\n--- Chunk {i+1} ---\n{chunk}")


