from google import generativeai as genai
import json
import os
from dotenv import load_dotenv

def analyze_chunk(client, chunk_data):
    """
    Analyzes a chunk and enriches it with a summary while maintaining existing fields.
    
    Args:
        client: The Google AI client
        chunk_data: A dictionary containing chunk information
    
    Returns:
        A dictionary with title, label, summary, and content fields
    """
    # Initialize the output dictionary with existing fields
    result = {
        "title": chunk_data.get("title", ""),
        "label": chunk_data.get("label", ""),
        "content": chunk_data.get("content", "")
    }
    
    # Only generate summary if not already present
    if "summary" not in chunk_data:
        try:
            # Set up the generation config
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 256,
            }
            
            # Get the Gemini model
            model = client.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            prompt = "Summarize this document"
            prompt = f"""
            Generate a brief summary (2-3 sentences) of the key information in this text. 
            Return ONLY the summary text, no additional commentary.
            
            Text to summarize:
            {chunk_data['content']}
            """
            
            # Generate summary
            response = model.generate_content(prompt)
            result["summary"] = response.text.strip()
            
            # Generate title if not present
            if not result["title"]:
                title_prompt = f"""
                Create a concise, descriptive title (5-10 words) that captures the main subject of this text.
                Return ONLY the title, nothing else.
                
                Text:
                {chunk_data['content']}
                """
                title_response = model.generate_content(title_prompt)
                result["title"] = title_response.text.strip()
                
        except Exception as e:
            result["summary"] = f"Error generating summary: {str(e)}"
            if not result["title"]:
                result["title"] = "Untitled Document"
    else:
        result["summary"] = chunk_data["summary"]
    
    return result

def process_chunks(chunks_data,apiKey):
    """
    Process each chunk in the input data
    
    Args:
        chunks_data: List of dictionaries containing chunk data
    
    Returns:
        List of processed chunks with title, label, summary, content
    """
    # Configure the API key

    genai.configure(api_key=apiKey)
    
    results = []
    for chunk in chunks_data:
        processed_chunk = analyze_chunk(genai, chunk)
        results.append(processed_chunk)
    
    return results

# Example usage
if __name__ == "__main__":
    # Read input JSON from a file (or parse from string)
    with open("rfp_chunks_labeled.json", "r") as f:
        input_chunks = json.load(f)
    api_key = os.getenv('API_KEY')
    # Process chunks
    processed_chunks = process_chunks(input_chunks,api_key)
    
    # Output the results as a formatted JSON
    print(json.dumps(processed_chunks, indent=2))
    
    # Save to a file
    with open("processed_chunks.json", "w") as f:
        json.dump(processed_chunks, f, indent=2)