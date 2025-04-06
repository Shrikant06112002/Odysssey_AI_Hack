from PreProcessing.Chunking import semantic_chunk_pdf_json
from PreProcessing.create_embedding import generate_embeddings_with_keywords
from PreProcessing.extractComData import extract_company_data
from Agents.compliance_check import run_compliance_check
from Agents.contractRisk import analyze_contract_risks
from Agents.mandatoryEligibility import extract_eligibility_criteria
from Agents.submissionCheck import generate_submission_checklist

import json

if __name__ == "__main__":
    pdf_path = r"D:/OdysseyCode/Odysssey_AI_Hack/Dataset/ELIGIBLE RFP - 1.pdf"
    docx_path = "D:/OdysseyCode/Odysssey_AI_Hack/Dataset/Company Data.docx"
    output_path = "chunked_output.json"
    # print("creating chunks for RFD")
    # chunks = semantic_chunk_pdf_json(pdf_path)
    # print("genrerating embeddings")
    # embeddings = generate_embeddings_with_keywords(chunks)
    print("extract_company_data")
    result = extract_company_data(docx_path)
    dicDataCom = json.dumps(result, indent=4)
    compliance_check = run_compliance_check(result)
    print("compliance_check",compliance_check)
    eligibility_criteria = extract_eligibility_criteria(result)
    print("eligibility_criteria",eligibility_criteria)
    submission_checklist = generate_submission_checklist()
    print("submission_checklist",submission_checklist)
    analyze_contract = analyze_contract_risks(result)
    print("analyze_contract",analyze_contract)
    

    
    # Pretty print the result
    # import json
    # print(json.dumps(result, indent=4))

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(chunks, f, indent=2, ensure_ascii=False)