from typing import List, Dict
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

# Single index for RFP documents
index_name = "eligibledocone"
index = pc.Index(index_name)

# Keywords focused on submission requirements
SUBMISSION_KEYWORDS = [
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

def generate_submission_checklist():
    """
    Extracts submission requirements from RFP documents and generates a structured checklist
    """
    # Create keyword query string for better retrieval
    keyword_query = " ".join(SUBMISSION_KEYWORDS)
    
    print("🔍 Embedding submission requirements query...")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']

    # Retrieve RFP document sections focused on submission requirements
    print("📄 Retrieving submission instruction sections from RFP...")
    submission_chunks = index.query(
        namespace="ns",
        vector=query_vector,
        top_k=5,
        include_values=False,
        include_metadata=True
    )

    # Build context with metadata
    rfp_context = "\n\n".join([
        f"Section ID: {match['id']}\n"
        f"Section Title: {match['metadata'].get('Section Title', 'N/A')}\n"
        f"Content: {match['metadata'].get('chunk', '')}\n"
        for match in submission_chunks['matches']
    ])
    
    # Prompt template focused on submission requirements extraction
    prompt_template = PromptTemplate(
        input_variables=["rfp_context"],
        template="""
You are an RFP response specialist tasked with creating a comprehensive submission checklist.

Review these RFP document sections containing submission instructions:
{rfp_context}

TASK:
Extract and structure ALL submission requirements into a detailed checklist that can be used by the proposal team.

Focus on identifying:
1. Document formatting requirements (page limits, font type/size, line spacing, margins, etc.)
2. Required attachments, forms, and templates
3. Submission methods and delivery instructions
4. Deadline information and timing requirements
5. Organization requirements (sections, table of contents, tabs, etc.)
6. Any special instructions that might disqualify the proposal if not followed

Return this JSON format:
{{
  "formatting_requirements": [
    {{
      "requirement_type": "Page Limit|Font|Spacing|Margins|Headers/Footers|Other",
      "description": "The specific requirement details",
      "section_reference": "Section where this requirement was found",
      "notes": "Additional clarification if needed"
    }}
  ],
  "required_attachments": [
    {{
      "attachment_name": "Name of the form/attachment",
      "description": "What this attachment contains",
      "mandatory": true|false,
      "special_instructions": "Any specific instructions for this attachment"
    }}
  ],
  "submission_instructions": [
    {{
      "instruction_type": "Method|Deadline|Copies|Packaging|Other",
      "description": "The specific instruction details",
      "notes": "Additional clarification if needed"
    }}
  ],
  "organization_requirements": [
    {{
      "requirement_type": "Table of Contents|Section Order|Tabs|Other",
      "description": "The specific requirement details",
      "notes": "Additional clarification if needed"
    }}
  ],
  "disqualification_triggers": [
    "List of specific items that would disqualify the submission"
  ],
  "submission_checklist_summary": "Brief summary of key requirements for quick reference"
}}
"""
    )

    prompt = prompt_template.format(rfp_context=rfp_context)

    # Run Gemini LLM
    print("🧠 Extracting submission requirements and generating checklist...")

    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 1500
        }
    )

    response = model.generate_content(prompt)
    

    result = response.text
    print("✅ Submission checklist generation completed")
    return result

def search_for_templates():
    """
    Specifically searches for mentions of required templates in the RFP
    """
    template_keywords = ["template", "form", "attachment", "exhibit", "appendix", "required document"]
    keyword_query = " ".join(template_keywords)
    
    print("🔍 Searching for specific templates and forms...")
    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[keyword_query],
        parameters={"input_type": "query"}
    )

    query_vector = embedding[0]['values']
    
    template_chunks = index.query(
        namespace="ns",
        vector=query_vector,
        top_k=5,
        include_values=False,
        include_metadata=True
    )
    
    # Process template mentions
    template_context = "\n\n".join([
        f"Section ID: {match['id']}\n"
        f"Section Title: {match['metadata'].get('Section Title', 'N/A')}\n"
        f"Content: {match['metadata'].get('chunk', '')}\n"
        for match in template_chunks['matches']
    ])
    
    # Extract template information
    prompt_template = PromptTemplate(
        input_variables=["template_context"],
        template="""
Extract information about ALL required templates, forms, and attachments mentioned in these RFP sections:

{template_context}

Return only a JSON array of objects — no extra text, no headers, no code blocks.
Extract details of all templates/forms mentioned in the input.
[
  {
    "template_name": "The name or identifier of the template/form",
    "purpose": "What this template/form is used for",
    "location": "Where in the RFP this template can be found (section/page/appendix)",
    "format": "The format of the template (Word, Excel, PDF, etc.)",
    "instructions": "Any specific instructions for completing this template"
  }
]


Only include actual templates/forms/attachments, not general document requirements.
"""
    )
    
    prompt = prompt_template.format(template_context=template_context)
    
    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_output_tokens": 800
        }
    )
    
    response = model.generate_content(prompt)
    
    templates = response.text
    return templates

def generate_comprehensive_checklist():
    """
    Generates a comprehensive submission checklist with additional template information
    """
    # Get base checklist
    checklist = generate_submission_checklist()
    
    # Augment with specific template information
    templates = search_for_templates()
    
    # If templates were successfully found, add them to the checklist
    if templates and isinstance(templates, list) and len(templates) > 0:
        # Create a detailed templates section
        checklist["templates_and_forms_detail"] = templates
    
    # Generate a printable version of the checklist
    checklist_printable = generate_printable_checklist(checklist)
    checklist["printable_checklist"] = checklist_printable
    
    return checklist

def generate_printable_checklist(checklist_data):
    """
    Converts the JSON checklist into a printable markdown format
    """
    prompt_template = PromptTemplate(
        input_variables=["checklist_json"],
        template="""
Convert this JSON checklist data into a well-formatted, user-friendly markdown checklist that can be printed:
Note:- Note give any extra out like ```json`` or anything else, just return the JSON
{checklist_json}


Format it as a proper checklist with checkboxes ([ ]) that can be checked off, clear headings, and logical organization.
Include all the important details but make it concise and practical for proposal teams to use.
"""
    )
    
    prompt = prompt_template.format(checklist_json=json.dumps(checklist_data))
    
    model = GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_output_tokens": 1000
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

        # Clean up markdown code block if present
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1:]
            
            if content.endswith("```"):
                content = content[:-3].strip()
            elif "```" in content:
                content = content[:content.rfind("```")].strip()

        # Parse JSON
        result = json.loads(content)
        return result

    except Exception as e:
        return response.text.strip()

    