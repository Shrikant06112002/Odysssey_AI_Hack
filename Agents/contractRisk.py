from typing import List
import json
from google import generativeai as genai
from google.generativeai import GenerativeModel, configure
from langchain.prompts import PromptTemplate
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemani_api_key = os.getenv("API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
client = genai.configure(api_key=gemani_api_key)

# Load your single vector database for contract documents
index_name = "eligibledocone"  # Single index containing contract documents
index = pc.Index(index_name)

# Keywords focused on potentially risky contract clauses
RISK_KEYWORDS = [
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
    "Intellectual property"
]

def analyze_contract_risks(COMPANY_DATA: dict) :
    # Create keyword query string for better retrieval
    keyword_query = " ".join(RISK_KEYWORDS)
    
    print("🔍 Embedding contract risk query...")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']

    # Retrieve contract document sections focused on risk areas
    print("📄 Retrieving high-risk sections from contract...")
    risk_chunks = index.query(
        namespace="ns",
        vector=query_vector,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

    # Build context with metadata
    contract_context = "\n\n".join([
        f"Section ID: {match['id']}\n"
        f"Section Title: {match['metadata'].get('Section Title', 'N/A')}\n"
        f"Content: {match['metadata'].get('chunk', '')}\n"
        for match in risk_chunks['matches']
    ])
    
    # Format company data for the prompt
    company_data_formatted = "\n".join([f"{key}: {value}" for key, value in COMPANY_DATA.items()])
    
    # Prompt template focused on contract risk analysis - limited to 2-3 sample clauses
    prompt_template = PromptTemplate(
        input_variables=["contract_context", "company_data"],
        template="""
You are an expert legal contract analyzer helping ConsultAdd identify unfavorable contract terms.

Review these contract document sections:
{contract_context}

Consider ConsultAdd's profile:
{company_data}

TASKS:
1. Identify ONLY the 2-3 most important biased or unfavorable clauses that put ConsultAdd at a disadvantage
2. Analyze the risk level of each problematic clause (Low, Medium, High)
3. Suggest specific modifications to balance each clause while remaining reasonable 
4. Provide reasoning for why the suggested modifications would better protect ConsultAdd

Focus on clauses related to: termination rights, liability limitations, payment terms, IP ownership, 
indemnification, warranties, and exclusivity requirements.

Return only pure JSON — no code blocks, no markdown, no extra text.
Identify exactly 3 clauses from the contract that may be biased or disadvantageous to ConsultAdd.
{{{{ 
  "biased_clauses": [
    {{
      "section_id": "The section ID from the contract",
      "clause_text": "The specific clause text that is concerning",
      "issue": "Brief description of how this clause disadvantages ConsultAdd",
      "risk_level": "Low|Medium|High",
      "recommendation": "Specific suggested modification to balance the clause",
      "reasoning": "Why this modification would better protect ConsultAdd's interests"
    }}
  ],
  "priority_concerns": [
    "List of 2-3 highest priority issues that should be addressed first"
  ],
  "overall_assessment": "Brief assessment of the contract's overall balance and key negotiation points"
}}}}
"""
    )

    prompt = prompt_template.format(
        contract_context=contract_context,
        company_data=company_data_formatted
    )

    # Run Gemini LLM
    print("🧠 Analyzing contract risks and identifying biased clauses...")

    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest", #gemini-1.5-pro-latest
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 1500
        }
    )

    response = model.generate_content(prompt)
    

        # Parse the JSON response
    result = response.text
    print("✅ Contract risk analysis completed")
    return result

def generate_balanced_clause(original_clause, clause_type):
    """
    Generates a balanced alternative clause based on the original problematic clause
    and reference sections from the same contract
    """
    print(f"✍️ Generating balanced alternative for {clause_type} clause...")
    
    # Get reference sections using specific legal terms related to this clause type
    reference_keywords = [clause_type, "fair", "balanced", "mutual", "reasonable"]
    
    # Create embedding for the combined keywords
    keyword_query = " ".join(reference_keywords)
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']
    
    # Get additional relevant sections from the same index
    reference_data = index.query(
        namespace="ns",
        vector=query_vector,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    
    reference_context = "\n\n".join([match['metadata'].get('chunk', '') for match in reference_data['matches']])
    
    # Prompt for generating a balanced clause
    prompt_template = PromptTemplate(
        input_variables=["original_clause", "clause_type", "reference_context", "company_data_formatted"],
        template="""
You are an expert legal contract drafter. Draft a balanced alternative to this problematic {clause_type} clause.

Original clause:
{original_clause}

Reference sections from the contract:
{reference_context}

Company context:
{company_data_formatted}

Create a balanced alternative clause that:
1. Protects ConsultAdd's interests while being fair to both parties
2. Uses clear, precise legal language
3. Addresses the core issues with the original clause
4. Would be reasonably acceptable to the counterparty

Return just the drafted clause text without additional comments.
"""
    )
    
    company_data_formatted = "\n".join([f"{key}: {value}" for key, value in COMPANY_DATA.items()])
    
    prompt = prompt_template.format(
        original_clause=original_clause,
        clause_type=clause_type,
        reference_context=reference_context,
        company_data_formatted=company_data_formatted
    )
    
    # Run Gemini LLM
    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 800
        }
    )
    response = model.generate_content(prompt)
    try:
        # Extract content from response
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

    
