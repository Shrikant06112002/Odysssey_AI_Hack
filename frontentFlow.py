import streamlit as st
import json
import os
import re
from PIL import Image
import pandas as pd

# Import your processing modules
from PreProcessing.Chunking import semantic_chunk_pdf_json
from PreProcessing.create_embedding import generate_embeddings_with_keywords
from PreProcessing.extractComData import extract_company_data
from Agents.compliance_check import run_compliance_check
from Agents.contractRisk import analyze_contract_risks
from Agents.mandatoryEligibility import extract_eligibility_criteria
from Agents.submissionCheck import generate_submission_checklist

# Set page config
st.set_page_config(page_title="RFP Analysis Tool", layout="wide")

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a86e8;
        color: white;
    }
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4a86e8;
    }
    .success {
        border-left: 5px solid #32CD32;
    }
    .warning {
        border-left: 5px solid #FFA500;
    }
    .danger {
        border-left: 5px solid #FF4500;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("RFP Analysis Tool")
st.write("Upload RFP documents and company data to analyze eligibility, compliance, and risks.")

# Function to safely parse JSON data
def parse_json_safely(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    if isinstance(data, str):
        # Remove backticks and "json" tag if present (for outputs like ```json {...} ```)
        cleaned_data = re.sub(r'^```json\s*|\s*```$', '', data.strip())
        try:
            return json.loads(cleaned_data)
        except json.JSONDecodeError:
            st.error(f"Could not parse JSON data: {cleaned_data[:100]}...")
            return {"error": "Could not parse JSON data"}
    return data if isinstance(data, dict) else {"error": "Data is not a dictionary"}

# Session state initialization
if 'company_data' not in st.session_state:
    st.session_state.company_data = None
if 'rfp_chunks' not in st.session_state:
    st.session_state.rfp_chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'compliance_check' not in st.session_state:
    st.session_state.compliance_check = None
if 'eligibility_criteria' not in st.session_state:
    st.session_state.eligibility_criteria = None
if 'submission_checklist' not in st.session_state:
    st.session_state.submission_checklist = None
if 'contract_risks' not in st.session_state:
    st.session_state.contract_risks = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# File Upload Section
st.header("Document Upload")
col1, col2 = st.columns(2)

with col1:
    st.subheader("RFP Document")
    rfp_file = st.file_uploader("Upload RFP PDF", type="pdf")
    
with col2:
    st.subheader("Company Data")
    company_file = st.file_uploader("Upload Company Data", type="docx")

# Process Files Button
if st.button("Process Documents", disabled=(rfp_file is None or company_file is None)):
    with st.spinner("Processing documents..."):
        # Save uploaded files temporarily
        temp_rfp_path = f"temp_rfp.pdf"
        temp_company_path = f"temp_company.docx"
        
        with open(temp_rfp_path, "wb") as f:
            f.write(rfp_file.getbuffer())
        
        with open(temp_company_path, "wb") as f:
            f.write(company_file.getbuffer())
        
        # Process RFP
        st.info("Creating semantic chunks from RFP...")
        # rfp_chunks = semantic_chunk_pdf_json(temp_rfp_path)
        # st.session_state.rfp_chunks = rfp_chunks
        
        st.info("Generating embeddings...")
        # embeddings = generate_embeddings_with_keywords(rfp_chunks)
        # st.session_state.embeddings = embeddings
        company_data = extract_company_data(temp_company_path)
        # Process Company Data
        st.info("Extracting company data...")
        
        st.session_state.company_data = parse_json_safely(company_data)
        
        # Run analysis
        st.info("Running compliance check...")
        try:
            compliance_check = run_compliance_check(company_data)
            st.session_state.compliance_check = compliance_check
        except Exception as e:
            st.session_state.compliance_check = compliance_check

        
        st.info("Extracting eligibility criteria...")
        try:
            eligibility_criteria = extract_eligibility_criteria(company_data)
            st.session_state.eligibility_criteria = eligibility_criteria
        except Exception as e:
            st.session_state.eligibility_criteria = eligibility_criteria


        st.info("Generating submission checklist...")
        try:
            submission_checklist = generate_submission_checklist()
            st.session_state.submission_checklist = submission_checklist
        except Exception as e:
            st.session_state.submission_checklist = submission_checklist


        st.info("Analyzing contract risks...")
        try:
            contract_risks = analyze_contract_risks(company_data)
            st.session_state.contract_risks = contract_risks
        except Exception as e:
            st.session_state.contract_risks = {}
        
        # Clean up temp files
        os.remove(temp_rfp_path)
        os.remove(temp_company_path)
        
        st.session_state.processing_complete = True
        st.success("Processing complete!")

# Just for debugging - display raw parsed data
if st.session_state.processing_complete and st.checkbox("Show Debug Raw Data"):
    st.subheader("Debug: Raw Parsed Data")
    st.json(st.session_state.compliance_check)
    st.json(st.session_state.eligibility_criteria)
    st.json(st.session_state.submission_checklist)
    st.json(st.session_state.contract_risks)

# Display Results if processing is complete
if st.session_state.processing_complete:
    st.header("Analysis Results")
    
    tabs = st.tabs(["Overview", "Compliance", "Eligibility", "Submission Checklist", "Contract Risks", "Raw Data"])
    
    # Overview Tab
    with tabs[0]:
        st.subheader("RFP Analysis Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get data safely
        compliance_data = st.session_state.compliance_check
        eligibility_data = st.session_state.eligibility_criteria
        contract_risks_data = st.session_state.contract_risks
        checklist_data = st.session_state.submission_checklist
        
        # Calculate compliance score
        compliance_score = 0
        if isinstance(compliance_data, dict):
            # Try to calculate from checks array first
            if "checks" in compliance_data and isinstance(compliance_data["checks"], list):
                checks = compliance_data["checks"]
                if checks:
                    passed_checks = sum(1 for check in checks if isinstance(check, dict) and check.get("passed", False))
                    compliance_score = (passed_checks / len(checks)) * 100
            # If no valid checks or compliance_score is still 0, try to get direct compliance_score
            if compliance_score == 0 and "compliance_score" in compliance_data:
                try:
                    compliance_score = float(compliance_data["compliance_score"])
                except (ValueError, TypeError):
                    compliance_score = 0
            # If still 0, check if there's an is_eligible key
            if compliance_score == 0 and "is_eligible" in compliance_data:
                compliance_score = 100 if compliance_data["is_eligible"] else 0
        
        # Calculate eligibility score
        eligibility_score = 0
        if isinstance(eligibility_data, dict):
            # Try mandatory_criteria first (from your sample data)
            criteria_list = eligibility_data.get("mandatory_criteria", [])
            if not criteria_list and "criteria" in eligibility_data:
                criteria_list = eligibility_data["criteria"]
                
            if criteria_list and isinstance(criteria_list, list):
                # Count criteria that have 'has_requirement' set to true or 'meets_criteria' set to true
                valid_criteria = [c for c in criteria_list if isinstance(c, dict)]
                if valid_criteria:
                    passed_criteria = sum(1 for c in valid_criteria if 
                        (c.get("has_requirement") is True) or (c.get("meets_criteria") is True))
                    eligibility_score = (passed_criteria / len(valid_criteria)) * 100
        
        # Get risk level
        risk_level = "Unknown"
        if isinstance(contract_risks_data, dict):
            if "overall_risk_level" in contract_risks_data:
                risk_level = contract_risks_data["overall_risk_level"]
            elif "overall_assessment" in contract_risks_data:
                # Extract risk level from assessment if available
                assessment = contract_risks_data["overall_assessment"]
                if "high risk" in assessment.lower():
                    risk_level = "High"
                elif "medium risk" in assessment.lower():
                    risk_level = "Medium"
                elif "low risk" in assessment.lower():
                    risk_level = "Low"
        
        # Calculate checklist completion
        checklist_completion = 0
        if isinstance(checklist_data, dict):
            checklist_items = []
            # Check different possible keys based on your sample data
            if "checklist_items" in checklist_data:
                checklist_items = checklist_data["checklist_items"]
            elif "required_attachments" in checklist_data:
                checklist_items = checklist_data["required_attachments"]
            
            if checklist_items and isinstance(checklist_items, list):
                valid_items = [item for item in checklist_items if isinstance(item, dict)]
                if valid_items:
                    completed_items = sum(1 for item in valid_items if 
                        item.get("status") == "completed" or item.get("completed") is True)
                    checklist_completion = (completed_items / len(valid_items)) * 100
        
        # Display metrics
        col1.metric("Compliance Score", f"{compliance_score:.1f}%")
        col2.metric("Eligibility Score", f"{eligibility_score:.1f}%")
        col3.metric("Risk Level", risk_level)
        col4.metric("Checklist Completion", f"{checklist_completion:.1f}%")
        
        # Overall recommendation
        overall_recommendation = "Proceed with caution"
        if compliance_score > 60:
            overall_recommendation = "Recommended to Proceed"
            card_class = "success"
        elif compliance_score < 60 :
            overall_recommendation = "Not Recommended"
            card_class = "danger"
        else:
            card_class = "warning"
            
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h3>Overall Recommendation: {overall_recommendation}</h3>
            <p>Based on the analysis of compliance, eligibility criteria, and risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)

        # Add key insights section
        st.subheader("Key Insights")

        # Compliance insight
        if isinstance(compliance_data, dict):
            compliance_summary = compliance_data.get("summary", "No compliance summary available.")
            st.markdown(f"""
            <div class="result-card">
                <h4>Compliance</h4>
                <p>{compliance_summary}</p>
            </div>
            """, unsafe_allow_html=True)

        # Eligibility insight
        if isinstance(eligibility_data, dict):
            # Get summary if available, otherwise construct one
            eligibility_summary = eligibility_data.get("summary", "")
            if not eligibility_summary and "mandatory_criteria" in eligibility_data:
                criteria = eligibility_data["mandatory_criteria"]
                met_count = sum(1 for c in criteria if isinstance(c, dict) and c.get("has_requirement") is True)
                total_count = len(criteria)
                eligibility_summary = f"Company meets {met_count} out of {total_count} mandatory criteria."
            
            if not eligibility_summary:
                eligibility_summary = "Eligibility criteria analysis complete."

            st.markdown(f"""
            <div class="result-card">
                <h4>Eligibility</h4>
                <p>{eligibility_summary}</p>
            </div>
            """, unsafe_allow_html=True)

        # Contract risks insight
        if isinstance(contract_risks_data, dict):
            risk_summary = contract_risks_data.get("overall_assessment", "")
            if not risk_summary:
                risk_summary = "Contract risk analysis complete."

            st.markdown(f"""
            <div class="result-card">
                <h4>Contract Risks</h4>
                <p>{risk_summary}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Compliance Tab
    with tabs[1]:
        st.subheader("Compliance Analysis")
        
        compliance_data = st.session_state.compliance_check
        
        if isinstance(compliance_data, dict):
            # Display compliance score and summary
            compliance_summary = compliance_data.get('summary', 'No summary available')
            is_eligible = compliance_data.get('is_eligible', None)
            
            status_class = "success" if is_eligible else "danger" if is_eligible is False else "warning"
            status_text = "Eligible" if is_eligible else "Not Eligible" if is_eligible is False else "Unknown"
            
            st.markdown(f"""
            <div class="result-card {status_class}">
                <h3>Compliance Status: {status_text}</h3>
                <p>{compliance_summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display compliance checks
            st.subheader("Compliance Checks")
            checks = compliance_data.get('checks', [])
            compliance_issues = compliance_data.get('compliance_issues', [])
            
            if checks and isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                        
                    passed = check.get('passed', False)
                    card_class = "success" if passed else "danger"
                    area = check.get('area', 'Unnamed Check')
                    note = check.get('note', 'No details available')
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h4>{area}</h4>
                        <p><strong>Status:</strong> {'Passed' if passed else 'Failed'}</p>
                        <p>{note}</p>
                    </div>
                    """, unsafe_allow_html=True)
            elif compliance_issues and isinstance(compliance_issues, list):
                for issue in compliance_issues:
                    if not isinstance(issue, dict):
                        continue
                        
                    severity = issue.get('severity', 'Medium').lower() if isinstance(issue.get('severity'), str) else 'Medium'
                    card_class = "success" if severity == "low" else "warning" if severity == "medium" else "danger"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h4>{issue.get('title', 'Compliance Issue')}</h4>
                        <p><strong>Severity:</strong> {issue.get('severity', 'Medium')}</p>
                        <p>{issue.get('description', 'No description available')}</p>
                        <p><strong>Recommendation:</strong> {issue.get('recommendation', 'No recommendation available')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No detailed compliance information found.")
        else:
            st.error("Compliance data is not in the expected format.")
            st.write(compliance_data)
    
    # Eligibility Tab
    with tabs[2]:
        st.subheader("Eligibility Criteria")
        
        eligibility_data = st.session_state.eligibility_criteria
        
        if isinstance(eligibility_data, dict):
            # Display summary if available
            summary = eligibility_data.get('summary', '')
            if summary:
                st.markdown(f"""
                <div class="result-card">
                    <h3>Eligibility Summary</h3>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Try to get criteria list from different possible keys
            criteria_list = []
            if "mandatory_criteria" in eligibility_data:
                criteria_list = eligibility_data["mandatory_criteria"]
            elif "criteria" in eligibility_data:
                criteria_list = eligibility_data["criteria"]
            
            # Display criteria
            st.subheader("Criteria Details")
            if criteria_list and isinstance(criteria_list, list):
                for criterion in criteria_list:
                    if not isinstance(criterion, dict):
                        continue
                    
                    # Check for different possible keys based on your sample data
                    meets = False
                    if "has_requirement" in criterion:
                        meets = criterion["has_requirement"]
                    elif "meets_criteria" in criterion:
                        meets = criterion["meets_criteria"]
                    
                    # Skip "unknown" criteria if present
                    if meets == "unknown":
                        card_class = "warning"
                    else:
                        card_class = "success" if meets else "danger"
                    
                    requirement = criterion.get('requirement', '')
                    name = criterion.get('name', '')
                    title = name if name else requirement
                    if not title:
                        title = 'Unnamed Criterion'
                    
                    notes = criterion.get('notes', '')
                    description = criterion.get('description', '')
                    details = notes if notes else description
                    
                    status_text = "Meets Criteria" if meets is True else "Does Not Meet Criteria" if meets is False else "Unknown"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h4>{title}</h4>
                        <p><strong>Status:</strong> {status_text}</p>
                        <p><strong>Requirement:</strong> {requirement}</p>
                        <p>{details}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No eligibility criteria found.")
        else:
            st.error("Eligibility data is not in the expected format.")
            st.write(eligibility_data)
    
    # Submission Checklist Tab
    with tabs[3]:
        st.subheader("Submission Checklist")
        
        checklist_data = st.session_state.submission_checklist
        
        if isinstance(checklist_data, dict):
            # Show progress bar
            st.progress(checklist_completion / 100)
            st.write(f"Checklist Completion: {checklist_completion:.1f}%")
            
            # Try to get description from different possible keys
            description = checklist_data.get('description', '')
            deadline = ''
            
            # Try to get deadline from submission instructions
            submission_instructions = checklist_data.get('submission_instructions', [])
            if submission_instructions and isinstance(submission_instructions, list):
                for instruction in submission_instructions:
                    if isinstance(instruction, dict) and instruction.get('instruction_type') == 'Deadline':
                        deadline = instruction.get('description', '')
                        break
            
            # Display submission requirements
            if description or deadline:
                st.markdown(f"""
                <div class="result-card">
                    <h3>Submission Requirements</h3>
                    {f"<p>{description}</p>" if description else ""}
                    {f"<p><strong>Deadline:</strong> {deadline}</p>" if deadline else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Display formatting requirements
            formatting_reqs = checklist_data.get('formatting_requirements', [])
            if formatting_reqs and isinstance(formatting_reqs, list):
                st.subheader("Formatting Requirements")
                for req in formatting_reqs:
                    if not isinstance(req, dict):
                        continue
                    
                    req_type = req.get('requirement_type', '')
                    desc = req.get('description', '')
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{req_type}</h4>
                        <p>{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Get checklist items from different possible keys
            checklist_items = []
            if "checklist_items" in checklist_data:
                checklist_items = checklist_data["checklist_items"]
            elif "required_attachments" in checklist_data:
                checklist_items = checklist_data["required_attachments"]
            
            # Display checklist items
            if checklist_items:
                st.subheader("Required Documents")
                
                for item in checklist_items:
                    if not isinstance(item, dict):
                        continue
                    
                    # Get name/title from different possible keys
                    name = item.get('name', '')
                    if not name:
                        name = item.get('attachment_name', 'Unnamed Item')
                    
                    # Get description
                    desc = item.get('description', 'No description available')
                    
                    # Get status
                    status = item.get('status', 'pending')
                    if status == 'completed':
                        card_class = "success"
                    else:
                        card_class = "warning"
                    
                    # Get additional instructions
                    special_instructions = item.get('special_instructions', '')
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h4>{name}</h4>
                        <p><strong>Status:</strong> {status.capitalize()}</p>
                        <p>{desc}</p>
                        {f"<p><strong>Special Instructions:</strong> {special_instructions}</p>" if special_instructions else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No checklist items found.")
            
            # Display submission instructions
            if submission_instructions:
                st.subheader("Submission Instructions")
                for instruction in submission_instructions:
                    if not isinstance(instruction, dict):
                        continue
                    
                    inst_type = instruction.get('instruction_type', '')
                    desc = instruction.get('description', '')
                    notes = instruction.get('notes', '')
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{inst_type}</h4>
                        <p>{desc}</p>
                        {f"<p><em>Note: {notes}</em></p>" if notes else ""}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Checklist data is not in the expected format.")
            st.write(checklist_data)
    
    # Contract Risks Tab
    with tabs[4]:
        st.subheader("Contract Risk Analysis")
        
        risk_data = st.session_state.contract_risks
        
        if isinstance(risk_data, dict):  
            # Get risk level and assessment
            risk_level = risk_data.get('overall_risk_level', 'Unknown')
            assessment = risk_data.get('overall_assessment', '')
            
            # Determine risk class
            risk_class = "success"
            if isinstance(risk_level, str):
                risk_class = "success" if risk_level.lower() in ["low"] else "danger" if risk_level.lower() in ["high", "critical"] else "warning"
            
            # Display overall assessment
            st.markdown(f"""
            <div class="result-card {risk_class}">
                <h3>Overall Risk Level: {risk_level}</h3>
                <p>{assessment}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get risk factors/biased clauses from different possible keys
            risk_factors = []
            if "risk_factors" in risk_data:
                risk_factors = risk_data["risk_factors"]
            elif "biased_clauses" in risk_data:
                risk_factors = risk_data["biased_clauses"]
            
            # Display priority concerns if available
            priority_concerns = risk_data.get('priority_concerns', [])
            if priority_concerns and isinstance(priority_concerns, list):
                st.subheader("Priority Concerns")
                for i, concern in enumerate(priority_concerns):
                    st.markdown(f"- **{concern}**")
            
            # Display risk factors
            if risk_factors and isinstance(risk_factors, list):
                st.subheader("Risk Factors")
                
                for factor in risk_factors:
                    if not isinstance(factor, dict):
                        continue
                    
                    # Get severity and name
                    severity = factor.get('severity', '')
                    if isinstance(severity, str):
                        severity = severity.lower()
                        card_class = "success" if severity == "low" else "danger" if severity in ["high", "critical"] else "warning"
                    else:
                        card_class = "warning"
                    
                    name = factor.get('name', '')
                    if not name:
                        name = factor.get('section_id', 'Risk Factor')
                    
                    # Get description
                    description = factor.get('description', '')
                    if not description:
                        description = factor.get('issue', 'No description available')
                    
                    # Get recommendation and mitigation
                    recommendation = factor.get('recommendation', '')
                    mitigation = factor.get('mitigation', '')
                    advice = recommendation if recommendation else mitigation
                    
                    # Get clause text if available
                    clause_text = factor.get('clause_text', '')
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <h4>Section {name}</h4>
                        <p><strong>Severity:</strong> {factor.get('severity', 'Medium')}</p>
                        {f"<p><strong>Clause:</strong> <em>{clause_text}</em></p>" if clause_text else ""}
                        <p><strong>Issue:</strong> {description}</p>
                        {f"<p><strong>Recommendation:</strong> {advice}</p>" if advice else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No risk factors found.")
        else:
            st.error("Risk data is not in the expected format.")
            st.write(risk_data)
    
    # Raw Data Tab
    with tabs[5]:
        st.subheader("Raw JSON Data")
        
        with st.expander("Company Data"):
            if isinstance(st.session_state.company_data, str):
                st.text(st.session_state.company_data)
            else:
                st.json(st.session_state.company_data)
        
        with st.expander("Compliance Check"):
            if isinstance(st.session_state.compliance_check, str):
                st.text(st.session_state.compliance_check)
            else:
                st.json(st.session_state.compliance_check)
        
        with st.expander("Eligibility Criteria"):
            if isinstance(st.session_state.eligibility_criteria, str):
                st.text(st.session_state.eligibility_criteria)
            else:
                st.json(st.session_state.eligibility_criteria)
        
        with st.expander("Submission Checklist"):
            if isinstance(st.session_state.submission_checklist, str):
                st.text(st.session_state.submission_checklist)
            else:
                st.json(st.session_state.submission_checklist)
        
        with st.expander("Contract Risks"):
            if isinstance(st.session_state.contract_risks, str):
                st.text(st.session_state.contract_risks)
            else:
                st.json(st.session_state.contract_risks)
            
        with st.expander("RFP Chunks Sample (First 3)"):
            if st.session_state.rfp_chunks:
                if isinstance(st.session_state.rfp_chunks, str):
                    try:
                        chunks = json.loads(st.session_state.rfp_chunks)
                        st.json(chunks[:3] if len(chunks) > 3 else chunks)
                    except:
                        st.text(st.session_state.rfp_chunks[:1000] + "..." if len(st.session_state.rfp_chunks) > 1000 else st.session_state.rfp_chunks)
                else:
                    st.json(st.session_state.rfp_chunks[:3] if len(st.session_state.rfp_chunks) > 3 else st.session_state.rfp_chunks)
            else:
                st.info("No RFP chunks available")

# Footer
st.markdown("---")
st.markdown("© 2025 RFP Analysis Tool | Powered by AI")