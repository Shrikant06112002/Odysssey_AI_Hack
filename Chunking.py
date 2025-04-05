from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("ELIGIBLE RFP - 1.pdf")
documents = loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,         # number of characters
    chunk_overlap=200,       # helps preserve context across chunks
)

chunks = text_splitter.split_documents(documents)

import json

# Convert chunks to JSON-serializable format
chunk_json = [
    {
        "chunk_id": i + 1,
        "content": chunk.page_content,
        "metadata": chunk.metadata
    }
    for i, chunk in enumerate(chunks)
]

# Print as formatted JSON
print(json.dumps(chunk_json, indent=2))