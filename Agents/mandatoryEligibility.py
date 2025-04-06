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

# Keywords focused on mandatory eligibility criteria
ELIGIBILITY_KEYWORDS = [
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
    "Registration"
]

def extract_eligibility_criteria():
    # Create keyword query string for better retrieval
    keyword_query = " ".join(ELIGIBILITY_KEYWORDS)
    
    print("🔍 Embedding eligibility criteria query...")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']

    # Retrieve RFP document sections focused on requirements
    print("📄 Retrieving eligibility sections from RFP...")
    eligibility_chunks = index_input.query(
        namespace="ns",
        vector=query_vector,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

    # Build context with metadata
    rfp_context = "\n\n".join([
        f"Section ID: {match['id']}\n"
        f"Section Title: {match['metadata'].get('Sub Title', 'N/A')}\n"
        f"Content: {match['metadata'].get('chunk', '')}\n"
        for match in eligibility_chunks['matches']
    ])
    
    # Format company data for the prompt
    company_data_formatted = "\n".join([f"{key}: {value}" for key, value in COMPANY_DATA.items()])
    
    # Prompt template focused on eligibility criteria extraction
    prompt_template = PromptTemplate(
        input_variables=["rfp_context", "company_data"],
        template="""
You are helping to extract all mandatory eligibility criteria from an RFP document.

Review these RFP document sections:
{rfp_context}

Compare with FirstStaff's information:
{company_data}

TASKS:
1. Extract ALL mandatory requirements (must-have qualifications, certifications, experience)
2. For each requirement, check if FirstStaff meets it based on their data
3. Flag any missing requirements that would prevent eligibility

Focus on phrases like "must have," "required," "minimum," and "mandatory."
Be practical in assessment - only flag truly missing requirements.

Return this JSON format:
{{
  "mandatory_criteria": [
    {{
      "requirement": "The specific requirement text",
      "category": "Qualification|Certification|Experience|Other",
      "has_requirement": true|false|unknown,
      "notes": "Brief note on whether FirstStaff meets this"
    }}
  ],
  "missing_requirements": [
    "List specific requirements FirstStaff is missing"
  ],
  "summary": "Brief summary of eligibility status based on criteria"
}}
"""
    )

    prompt = prompt_template.format(
        rfp_context=rfp_context,
        company_data=company_data_formatted
    )

    # Run Gemini LLM
    print("🧠 Extracting mandatory eligibility criteria...")

    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 1000
        }
    )

    response = model.generate_content(prompt)
    
    try:
        # Parse the JSON response
        result = json.loads(response.text)
        print("✅ Eligibility criteria extraction completed")
        return result
    except json.JSONDecodeError:
        # Handle case where response isn't valid JSON
        print("⚠️ Could not parse JSON response. Raw output:")
        print(response.text)
        return {"error": "Failed to parse response", "raw_response": response.text}
    
# Example usage
if __name__ == "__main__":
    result = extract_eligibility_criteria()
    
    print(json.dumps(result, indent=2))