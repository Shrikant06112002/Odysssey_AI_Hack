import fitz  # PyMuPDF
import re
import numpy as np
import tiktoken
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Load model and tokenizer once
embedder = SentenceTransformer('all-MiniLM-L6-v2')
encoding = tiktoken.get_encoding("cl100k_base")


CHECKPOINT_KEYWORDS = [
    "Key Areas for Innovation",
    "Automating Standard Compliance Checks",
    "legally eligible to bid",
    "state registration",
    "certifications",
    "past performance requirements",
    "deal-breakers",
    "Mandatory Eligibility Criteria",
    "must-have qualifications",
    "experience needed to bid",
    "missing requirements",
    "Submission Checklist",
    "RFP submission requirements",
    "Document format",
    "page limit",
    "font type",
    "font size",
    "line spacing",
    "TOC requirements",
    "attachments",
    "forms",
    "Contract Risks",
    "biased clauses",
    "unilateral termination rights",
    "notice period",
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

def merge_chunks(clusters: list[list[str]], max_tokens: int = 3000,overlap_tokens: int = 50) -> list[str]:
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

    return json_output

if __name__ == "__main__":
    pdf_path = r".\..\Dataset\ELIGIBLE RFP - 2.pdf"
    output_path = "chunked_output.json"
    chunks = semantic_chunk_pdf_json(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    # for i, chunk in enumerate(chunks):
    #     print(f"\n--- Chunk {i+1} ---\n{chunk}")


