from typing import List
from pinecone import Pinecone, ServerlessSpec
import time
import json
from google import generativeai as genai
from google.generativeai import GenerativeModel, configure
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemani_api_key = os.getenv("API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
client = genai.configure(api_key=gemani_api_key)
# Initialize your embeddings and LLM

index_name_ref = "refdocanalysis"
index_knowledge = pc.Index(index_name_ref)# Load your vector databases

index_name_ref = "userdocindex"
index_input = pc.Index(index_name_ref)# Load your vector databases
def run_compliance_check():
    query = "What are the legal, regulatory, or registration compliance requirements for vendors in this RFP?"

    # Step 1: Embed query
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']

    # Step 2: Retrieve top 3 chunks from user PDF
    top_pdf_chunks = index_input.query(
        namespace="ns1",
        vector=query_vector,
        top_k=1,
        include_values=False,
        include_metadata=True
    )

    # Build pdf context with metadata
    pdf_context = "\n\n".join([
        f"ID: {match['id']}\n"
        f"Sub Title: {match['metadata'].get('Sub Title', '')}\n"
        f"Chunk: {match['metadata'].get('chunk', '')}\n"
        f"Keywords: {match['metadata'].get('keywords', [])}"
        for match in top_pdf_chunks['matches']
    ])

    # Step 3: Retrieve 1 relevant chunk from compliance knowledge base
    compliance_kb_chunk = index_knowledge.query(
        namespace="ns1",
        vector=query_vector,
        top_k=1,
        include_values=False,
        include_metadata=True
    )

    # Build kb context with metadata
    kb_context = "\n\n".join([
        f"ID: {match['id']}\n"
        f"Title: {match['metadata'].get('Title', '')}\n"
        f"Sub Title: {match['metadata'].get('Sub Title', '')}\n"
        f"Context: {match['metadata'].get('Context', '')}\n"
        f"Chunk: {match['metadata'].get('chunk', '')}\n"
        f"Keywords: {match['metadata'].get('keywords', [])}"
        for match in compliance_kb_chunk['matches']
    ])

    # Step 4: Prompt template
    prompt_template = PromptTemplate(
        input_variables=["pdf_chunks", "compliance_kb"],
        template="""
You are a government contracting compliance assistant helping a company (ConsultAdd) assess whether they are eligible to respond to a government RFP (Request for Proposal).

Below are extracted chunks from a user-submitted RFP document, along with any relevant metadata:

--- RFP Document Chunks ---
{pdf_chunks}

Below is internal compliance knowledge, derived from past submissions and legal requirements:

--- Internal Compliance Knowledge ---
{compliance_kb}

TASK:
- Analyze the compliance requirements from the RFP content.
- Determine if ConsultAdd is eligible to bid, based on standard rules in the knowledge section.
- Highlight any missing certifications, registrations, or experience.
- Flag any deal-breakers that would make ConsultAdd ineligible to apply.

Respond strictly in this JSON format:

{{
  "is_eligible": true | false,
  "missing_requirements": ["..."],
  "deal_breakers": ["..."],
  "summary": "..."
}}
"""
    )

    prompt = prompt_template.format(
        pdf_chunks=pdf_context,
        compliance_kb=kb_context
    )

    # Step 5: Run Gemini LLM
    print("🧠 Running LLM on compliance check...")

    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 512
        }
    )

    response = model.generate_content(prompt)

    # Step 6: Show the response
    print("✅ Compliance Check Result:")
    print(response.text)

