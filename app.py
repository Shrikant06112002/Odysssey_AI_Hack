from PreProcessing.Chunking import semantic_chunk_pdf_json
from PreProcessing.create_embedding import generate_embeddings_with_keywords
from Agents.compliance_check import run_compliance_check
import json

if __name__ == "__main__":
    pdf_path = r"D:/OdysseyCode/Odysssey_AI_Hack/Dataset/ELIGIBLE RFP - 2.pdf"
    output_path = "chunked_output.json"
    
    chunks = semantic_chunk_pdf_json(pdf_path)
    embeddings = generate_embeddings_with_keywords(chunks)
    run_compliance_check()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)