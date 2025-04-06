# RFP-Analyzer: AI-Powered Government Contract Analysis

RFP-Analyzer automates the analysis of government Request for Proposals (RFPs) using Generative AI, Retrieval-Augmented Generation (RAG), and agentic workflows. It streamlines proposal preparation for ConsultAdd by automating eligibility checks, extracting requirements, and assessing risks.

---

## 🚀 Overview

Manual RFP analysis is often time-consuming, error-prone, and inconsistent. RFP-Analyzer addresses this with an AI-driven solution that:

- Verifies eligibility against company qualifications
- Extracts and structures RFP requirements
- Generates submission checklists
- Identifies contract risks and provides mitigation suggestions

---

## 🔍 Key Features

### 1. **Eligibility Assessment Engine**
- Evaluates if ConsultAdd meets RFP eligibility criteria
- Provides "Go/No-Go" recommendations with justification

### 2. **Requirement Extraction System**
- Converts unstructured RFP text into structured data
- Categorizes and prioritizes requirements by type and urgency

### 3. **Compliance Checklist Generator**
- Automatically creates custom submission checklists
- Captures formatting requirements, document templates, and required attachments

### 4. **Risk Analysis Module**
- Flags potentially problematic contract clauses
- Recommends balanced language alternatives

---

## 🧠 Technical Architecture

- **Frontend:** streamlit 
- **Backend:** Python   
- **AI/ML:** LLMs, RAG using Pinecone vector database, Agentic AI  
- **Workflow Management:** Agentic system with specialized agents for each analysis task

---

## 📦 RAG Implementation

- **Semantic Search:** Embedding-based document retrieval  
- **Knowledge Integration:** 
  - Internal company database for eligibility checks  
  - Legal precedent corpus for contract risk analysis  

---
