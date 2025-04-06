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
    "Registration",
    "Certification",
    "Experience",
    "Compliance",
    "Eligibility",
    "Audit",
    "Security",
    "Turnover",
    "GST",
    "Insurance",
    "Terminate",
    "Liability",
    "Indemnity",
    "Warranty",
    "Damages",
    "Penalty",
    "Unilateral",
    "Amendment",
    "Obligation",
    "Exclusive",
    "Jurisdiction",
    "Force Majeure",
    "Non-compete",
    "Payment terms",
    "Intellectual property",
    "Required",
    "Mandatory",
    "Must",
    "Minimum",
    "Qualification",
    "Criteria",
    "Eligibility",
    "Prerequisite",
    "Essential",
    "Experience",
    "Certification",
    "License",
    "Registration",
    "submit",
    "submission",
    "requirement",
    "format",
    "guideline",
    "instruction",
    "page limit",
    "page count",
    "font",
    "margin",
    "spacing",
    "attachment",
    "form",
    "deadline",
    "due date",
    "table of contents",
    "TOC",
    "appendix",
    "header",
    "footer",
    "binding",
    "electronic",
    "hard copy",
    "template"
 
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
    return json_output



