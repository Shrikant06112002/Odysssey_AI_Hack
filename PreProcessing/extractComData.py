from docx import Document
import re

def extract_company_data(docx_path):
    doc = Document(docx_path)
    data_dict = {}
    key = None

    def clean_text(text):
        return re.sub(r'\s+', ' ', text.strip())

    lines = []
    for para in doc.paragraphs:
        text = clean_text(para.text)
        if text:
            lines.append(text)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Case 1: Key followed by value in next line
        if (i + 1 < len(lines)) and not re.search(r'[:\-]', line) and not lines[i+1].startswith(('Key Personnel', 'Signature', 'Authorized Representative')):
            next_line = lines[i + 1]
            if next_line and not re.match(r'^[A-Z ]{3,}$', next_line):  # skip headers
                data_dict[line] = next_line
                i += 2
                continue

        # Case 2: Inline key-value pairs like "Company Legal Name: ABC Corp"
        if ':' in line:
            parts = re.split(r':\s*', line, maxsplit=1)
            if len(parts) == 2:
                k, v = map(clean_text, parts)
                data_dict[k] = v
                i += 1
                continue

        # Case 3: Handle Key Personnel separately
        if line.startswith("Key Personnel"):
            role = line.replace("Key Personnel –", "").strip()
            i += 1
            while i < len(lines) and not lines[i].startswith("Key Personnel"):
                person = clean_text(lines[i])
                if person:
                    data_dict[f"Key Personnel – {role}"] = person
                    break
                i += 1
            continue

        # If nothing matches, just move on
        i += 1

    return data_dict


# Example usage
if __name__ == "__main__":
    docx_path = ".\..\Dataset\Company Data.docx"  # replace with your file
    result = extract_company_data(docx_path)
    
    # Pretty print the result
    import json
    print(json.dumps(result, indent=4))
