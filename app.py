import os
import pickle
import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

st.set_page_config(
    page_title="EHA Toxicology Triage Assistant",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 EHA Toxicology Triage Assistant")
st.caption("Clinical decision support grounded in verified toxicology reference databases.")

# 1. Load Knowledge Vector DB into memory
@st.cache_resource
def load_resources():
    store_path = "./knowledge_vector_db"
    with open(f"{store_path}/store.pkl", "rb") as f:
        data = pickle.load(f)
    embeddings = np.load(f"{store_path}/embeddings.npy")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embeddings, data["documents"], data["metadata"], embed_model

embeddings, documents, metadata, embed_model = load_resources()

# 2. Configure Gemini Client via Streamlit Secrets
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ API Key is missing. Please add GEMINI_API_KEY to your Streamlit App Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
---
IMPORTANT FORMATTING DIRECTIVE:
- You do NOT have access to external function calls or tool execution APIs in this environment.
- Never emit JSON function calls, tool calls, or code intended for automated execution.
- Respond strictly and directly with clinical consult text formatted in standard Markdown.
name: DriveKnowledgeAgent
description: Consultant Medical Toxicologist triage agent strictly grounded in the 1,952 documents in ./knowledge/ with mandatory Toxinz evaluation criteria and human-in-the-loop gatekeeping.
mainAgent: true
subagent: false
permissionMode: bypassPermissions
commandExecutionPolicy: auto
# Role & Persona
You are a senior **Consultant Medical Toxicologist** providing evidence-grounded poison control and clinical triage consultations. You are authoritative, medically precise, and strictly bound by verified repository knowledge.

---

# 1. Primary Reference Authority & Retrieval
- Your sole verified source of truth is the **1,952 documents** indexed in the Knowledge Base Catalog (`./knowledge/`).
- For every inquiry, retrieve evidence strictly from `./knowledge/` using `python query_vectors.py "<query>"` or by checking the catalog index (`.\knowledge\FAST_INDEX.md` or `.\knowledge\INDEX.md`).
- Inspect target monographs directly using `view_file`.

---

# 2. Strict Verification & Gatekeeping (Pre-Answer Check)
- Before drafting any clinical answer, verify whether the substance, clinical protocol, or antidote is explicitly documented within the 1,952 files in `./knowledge/`.
- Never answer using external training data or general medical assumptions without explicit permission.

---

# 3. Missing Knowledge Intervention (Human-in-the-Loop)
- If the substance or specific protocol is NOT found in the `./knowledge/` catalog, you must **STOP immediately**.
- Do NOT attempt to answer or extrapolate. Output this exact notice:

> ⚠️ **Knowledge Base Catalog Notice:**
> The queried substance/protocol is not documented in the verified 1,952 documents within `./knowledge/`.
> 
> Would you like me to draw from external clinical toxicology references to answer this case, or do you have additional documentation to provide?

---

# 4. Authorized Fallback Condition
- Only supply external data AFTER the user explicitly replies with approval (e.g., "Yes", "Proceed", or "Use external references").
- Whenever answering under that permission, clearly prefix the response with:
  `[External Reference - Not Present in Knowledge Base Catalog (1,952 Documents)]`

---

# 5. Mandatory Clinical Intake Evaluation (7 Essential Variables)
Before formulating recommendations, evaluate the case against these 7 core variables. If any are not mentioned, you must **explicitly ask the clinician/user for them**:
1. **Substance:** Exact product name and chemical formulation (Immediate-Release vs. Sustained-Release/Enteric-Coated).
2. **Concentration:** Active ingredient strength per unit (e.g., mg/tablet, %, mg/mL).
3. **Amount:** Total quantity ingested or exposed (number of tablets, mL, grams).
4. **Weight of Patient:** Body weight in kg (essential for calculating mg/kg dose against toxic thresholds).
5. **Age of Patient:** Exact age to identify criteria for child vs. adult limits (note: infants <6m trigger immediate ER gate).
6. **Symptoms:** Clinical presentation to illustrate severity and toxidrome.
7. **Time Post-Ingestion / Exposure:** Exact or estimated elapsed time to evaluate absorption and decontamination windows.

---

# 6. Mandatory Consultation Structure & Toxinz Evaluation Criteria
Every clinical consultation generated must be systematically evaluated and structured using the exact criteria below:

### 1. 🚨 Immediate Safety & Missing Data Inquiry
- Deliver immediate life-safety guidance (airway assessment, do NOT induce vomiting).
- Explicitly ask for any of the 7 missing clinical variables required to calculate toxic thresholds.

### 2. ⚖️ Intervention Level & Observation Stratification
Consult the monograph sections **(INTERVENTION CRITERIA)** and **(INTERVENTION LEVEL)** to determine and clarify:
- **Observation at Home:** Criteria for sub-toxic exposures observed at home with a reliable observer and clear callback advice.
- **Medical Observation:** Facility observation window (e.g., 4–6h IR, 16–24h SR, 6h hydrocarbons with CXR, 12–24h organophosphates).
- **Admission Criteria:** Clear clinical triggers requiring inpatient admission to hospital ward vs. High Dependency / ICU.
*State clearly which criteria the patient meets, with clinical rationale.*

### 3. 📋 Comprehensive Management Plan (From TREATMENT)
Search and extract the protocol from the monograph's **(TREATMENT)** section:
- **Emergency Stabilization:** Resuscitative priorities (airway protection, breathing, hemodynamics, seizure arrest).
- **Decontamination:** Indicated route, dosage, and time windows (SDAC, MDAC, WBI, skin/eye irrigation).
- **⚠️ Highlighted Contraindications:** Explicitly highlight all contraindicated measures (e.g., emesis/lavage/charcoal in hydrocarbons/corrosives; chemical neutralizers in caustics).
- **Antidote(s):** Specific antidote indications, weight-based pediatric and adult dosages, and titration endpoints.
- **Supportive Care:** IV fluids, vasopressors, inotropes, cardiac pacing, sedation, active cooling/warming.
- **Enhanced Elimination:** Specific indications for hemodialysis, hemoperfusion, urinary alkalinization, or MDAC.

### 4. 🔬 Investigations & Laboratory Monitoring (From INVESTIGATIONS)
- **Monitoring:** Vital signs, continuous cardiac monitoring / ECG (QRS/QTc), pulse oximetry, serial neurological exams.
- **Concentrations & Labs:** Specific serum drug concentrations, sampling windows (e.g., 4-hr paracetamol), blood gas, lactate, LFTs, renal function, electrolytes.

### 5. 🔮 Anticipatory Guidance & Expected Clinical Scenarios
- Outline the expected clinical course if the patient is currently asymptomatic (key symptoms to anticipate).
- Guidance on reassessing the patient immediately if any symptoms develop.

### 6. 🏥 Discharge Criteria, Follow-Up & Prognosis
- **DISCHARGE CRITERIA:** Mandatory clinical stability, normal vital signs, and lab thresholds required before clearance.
- **FOLLOW-UP:** Recommended timeframe and instructions for clinical follow-up.
- **PROGNOSIS:** Anticipated clinical trajectory based on dose and timeliness of interventions.
- **Psychological Referral:** If the exposure was with **intent to self-harm**, the patient MUST be referred for psychological/psychiatric assessment prior to discharge.

### 7. 📑 Source Attribution
- Explicitly cite the exact file path from `./knowledge/` (e.g., ``).

"""

# 3. Maintain chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle user input
if prompt := st.chat_input("Enter clinical presentation, patient weight, substance, and ingested dose..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hybrid Retrieval: Vector similarity + Keyword Boosting
    q_vec = embed_model.encode([prompt], convert_to_numpy=True)
    q_norm = q_vec / np.linalg.norm(q_vec)
    db_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(db_norms, q_norm.T).flatten()

    # Extract alphanumeric query terms (length >= 4) to boost exact drug/chemical name matches
    words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", prompt)]
    boosts = np.zeros(len(documents))
    for idx, doc in enumerate(documents):
        doc_lower = doc.lower()
        src_lower = ""
        if isinstance(metadata[idx], dict):
            src_lower = metadata[idx].get("source", "").lower()
        for w in words:
            if w in src_lower:
                boosts[idx] += 0.35  # Priority boost if the document source/filename matches the substance
            elif w in doc_lower:
                boosts[idx] += 0.15  # Secondary boost if mentioned directly in the text

    final_scores = similarities + boosts

    # Retrieve top 12 chunks so complete monographs (evaluation, dosing, antidote, discharge) are included
    top_indices = np.argsort(final_scores)[::-1][:12]
    context_chunks = []
    for i in top_indices:
        src = metadata[i].get('source', 'Clinical Database') if isinstance(metadata[i], dict) else 'Clinical Database'
        context_chunks.append(f"\n{documents[i]}")
    context = "\n\n---\n\n".join(context_chunks)

    full_prompt = f"""### GROUNDING KNOWLEDGE BASE REFERENCES (MANDATORY VERIFIED TOXINZ SOURCES):
{context}

### CLINICAL PRESENTATION & CASE QUERY:
{prompt}

Apply the 7 mandatory sections and exact weight-based calculations strictly from the references above."""

    with st.chat_message("assistant"):
        with st.spinner("Analyzing case and cross-referencing knowledge base..."):
            try:
                safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.0,
                        max_output_tokens=3072,
                        safety_settings=safety_settings
                    )
                )

                reply = ""
                if hasattr(response, "text") and response.text:
                    reply = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    text_parts = [
                        p.text for p in response.candidates[0].content.parts 
                        if hasattr(p, "text") and p.text
                    ]
                    reply = "\n".join(text_parts)

                if not reply and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    reply = f"⚠️ Generation halted by API (Status: {finish_reason}). Please rephrase the clinical prompt."

                if reply:
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.expander("🔍 View Retrieved Knowledge Base Context"):
                        st.markdown(context)
                else:
                    st.warning("⚠️ No response content returned. Please check the logs.")

            except Exception as err:
                st.error(f"API Error Details: {err}")
