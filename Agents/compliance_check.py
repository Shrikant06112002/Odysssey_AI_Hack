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

# Load your vector databases
index_name_ref = "refdocanalysis"  # Knowledge base with compliance info
index_knowledge = pc.Index(index_name_ref)

index_name_input = "userdocindex"  # User uploaded RFP document
index_input = pc.Index(index_name_input)

# Define company data
COMPANY_DATA = {
    "Company Legal Name": "FirstStaff Workforce Solutions, LLC",
    "Principal Business Address": "3105 Maple Avenue, Suite 1200, Dallas, TX 75201",
    "Phone Number": "(214) 832-4455",
    "Fax Number": "(214) 832-4460",
    "Email Address": "proposals@firststaffsolutions.com",
    "Authorized Representative": "Meredith Chan, Director of Contracts",
    "Authorized Representative Phone": "(212) 555-0199",
    "Signature": "Meredith Chan (signed manually)",
    "Company Length of Existence": "9 years",
    "Years of Experience in Temporary Staffing": "7 years",
    "DUNS Number": "07-842-1490",
    "CAGE Code": "8J4T7",
    "SAM.gov Registration Date": "03/01/2022",
    "NAICS Codes": "561320 – Temporary Help Services; 541611 – Admin Management",
    "State of Incorporation": "Delaware",
    "Bank Letter of Creditworthiness": "Not Available",
    "State Registration Number": "SRN-DE-0923847",
    "Services Provided": "Administrative, IT, Legal & Credentialing Staffing",
    "Business Structure": "Limited Liability Company (LLC)",
    "W-9 Form": "Attached (TIN: 47-6392011)",
    "Certificate of Insurance": ""
}

# Define checkpoint keywords for compliance checks
CHECKPOINT_KEYWORDS = [
    "Registration",
    "Certification",
    "Experience",
    "Compliance",
    "Eligibility",
    "Audit",
    "Security",
    "Turnover",
    "GST"
]

def run_compliance_check():
    # Create keyword query string for better retrieval
    keyword_query = " ".join(CHECKPOINT_KEYWORDS)
    
    print("🔍 Embedding compliance query...")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']

    # Retrieve more chunks from the RFP document to ensure we capture all requirements
    print("📄 Retrieving relevant RFP sections...")
    top_pdf_chunks = index_input.query(
        namespace="ns",
        vector=query_vector,
        top_k=3,  # Increased to retrieve more relevant sections
        include_values=False,
        include_metadata=True
    )

    # Build pdf context with metadata
    pdf_context = "\n\n".join([
        f"Section ID: {match['id']}\n"
        f"Section Title: {match['metadata'].get('Sub Title', 'N/A')}\n"
        f"Content: {match['metadata'].get('chunk', '')}\n"
        f"Keywords: {match['metadata'].get('keywords', [])}"
        for match in top_pdf_chunks['matches']
    ])
    print("PDF Context:")
    # print(pdf_context)

    # Format company data for the prompt
    company_data_formatted = "\n".join([f"{key}: {value}" for key, value in COMPANY_DATA.items()])
    
    # More focused prompt template for compliance checking
    prompt_template = PromptTemplate(
        input_variables=["pdf_context", "company_data"],
                template="""
You are doing a quick preliminary check if FirstStaff Workforce Solutions is eligible to bid on an RFP.

Review these RFP document chunks:
{pdf_context}

Company information:
{company_data}

Do a BASIC check for ONLY these 4 items:
1. Legal registration (state registration)
2. Basic certifications (DUNS, CAGE, SAM.gov)
3. Past performance (years of experience)
4. Any obvious deal-breakers

Be LENIENT in your assessment. If there's any evidence the company meets a requirement or the requirement isn't clearly specified, consider it satisfied.

Return this simple JSON format:
{{
  "is_eligible": true|false,
  "checks": [
    {{
      "area": "Legal Registration",
      "passed": true|false,
      "note": "Brief note (1-2 sentences)"
    }},
    {{
      "area": "Certifications",
      "passed": true|false,
      "note": "Brief note (1-2 sentences)"
    }},
    {{
      "area": "Past Performance",
      "passed": true|false,
      "note": "Brief note (1-2 sentences)"
    }},
    {{
      "area": "Deal-Breakers",
      "passed": true|false,
      "note": "Brief note (1-2 sentences)"
    }}
  ],
  "summary": "One sentence summary"
}}
"""
    )

    prompt = prompt_template.format(
        pdf_context=pdf_context,
        company_data=company_data_formatted
    )

    # Run Gemini LLM with structured output format
    print("🧠LLM cooking compliance analysis...")

    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.2,  # Lower temperature for more factual output
            "top_p": 0.95,
            "max_output_tokens": 1500  # Increased token limit for more detailed analysis
        }
    )

    response = model.generate_content(prompt)
    
    # Parse the JSON response
    # result = json.loads(response.text)
    # print(response.text)
    print("✅ Compliance check completed successfully")
    return response.text

    
# Example usage
if __name__ == "__main__":
    result = run_compliance_check()
    
    print(result)