from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load the eligible and ineligible RFP PDFs
eligible_loader = PyPDFLoader("ELIGIBLE RFP - 1.pdf")
ineligible_loader = PyPDFLoader("IN-ELIGIBLE_RFP.pdf")

eligible_documents = eligible_loader.load()
ineligible_documents = ineligible_loader.load()

# Set up the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=150,
)

# Split documents into chunks
eligible_chunks = text_splitter.split_documents(eligible_documents)
ineligible_chunks = text_splitter.split_documents(ineligible_documents)

# Add labels and convert to JSON format
eligible_chunks_json = [
    {
        "chunk_id": i + 1,
        "label": "eligible",
        "content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for i, chunk in enumerate(eligible_chunks)
]

ineligible_chunks_json = [
    {
        "chunk_id": i + 1,
        "label": "ineligible",
        "content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for i, chunk in enumerate(ineligible_chunks)
]

# Combine both sets of chunks
combined_chunks = eligible_chunks_json + ineligible_chunks_json

# Save to a JSON file
with open("rfp_chunks_labeled.json", "w", encoding="utf-8") as f:
    json.dump(combined_chunks, f, indent=2)

print("✅ Chunking completed and saved to rfp_chunks_labeled.json")