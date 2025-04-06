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

def extract_eligibility_criteria(COMPANY_DATA: dict) :
    """_summary_

    Args:
        COMPANY_DATA (dict): _description_
    """   
    
    keyword_query = " ".join(ELIGIBILITY_KEYWORDS)
    index_name_input = "eligibledocone"  # User uploaded RFP document
    index_input = pc.Index(index_name_input)
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
Extract eligibility information from this RFP document and format it as JSON:
Review these RFP document sections: {rfp_context}
Compare with FirstStaff's information: {company_data}
Return ONLY this exact JSON format with no additional text:
{{ "is_eligible": true|false, "checks": [ {{ "area": "Qualification", "passed": true|false, "note": "Brief note (1-2 sentences)" }}, {{ "area": "Experience", "passed": true|false, "note": "Brief note (1-2 sentences)" }}, {{ "area": "Certification", "passed": true|false, "note": "Brief note (1-2 sentences)" }} ], "summary": "One sentence summary" }}

Focus on mandatory requirements with phrases like "must have," "required," "minimum," and "mandatory."
"""
    )

    prompt = prompt_template.format(
        rfp_context=rfp_context,
        company_data=company_data_formatted
    )
    index_name = "eligibledocone"  # Single index containing contract documents
    index = pc.Index(index_name)
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
        if hasattr(response, 'text'):
            content = response.text
        elif hasattr(response, 'parts') and len(response.parts) > 0:
            content = response.parts[0].text
        else:
            # Direct access for newer Gemini API versions
            content = str(response.candidates[0].content.parts[0].text)

        # Clean up the content by removing markdown code block markers
        if content.startswith("```"):
            # Find the first newline to skip the ```json line
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            
            # Remove the closing ``` if present
            if content.endswith("```"):
                content = content[:-3].strip()
            elif "```" in content:
                # In case there are trailing characters after the closing ```
                content = content[:content.rfind("```")].strip()

        # Parse the JSON response
        result = json.loads(content)
        return result
    except Exception as e:
        return response.text.strip()
        

    