import os
import pickle
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
You are a senior Consultant Medical Toxicologist triage officer.

Evaluate cases using this structured framework:
1. Resuscitation & Immediate Danger Signs (Airway, vitals, toxic toxidromes).
2. Toxicokinetic Assessment (Weight-based dose in mg/kg vs toxicity threshold, absorption window, trade name to generic active ingredient conversion like Cipralex -> Escitalopram).
3. Targeted Management & Antidote Protocol (Decontamination window, specific antidote, laboratory workup, observation timeframe).

GROUNDING RULES:
- Use the provided knowledge base excerpts to formulate your clinical advice.
- Recognize brand and trade names (e.g., Cipralex = escitalopram, Panadol = acetaminophen/paracetamol) and apply the corresponding management protocols from the excerpts.
- If and ONLY if the substance or class is entirely absent from the excerpts and clinical context, state the notice asking to use external references.
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

    # Retrieval: Vector similarity search across local database
    q_vec = embed_model.encode([prompt], convert_to_numpy=True)
    q_norm = q_vec / np.linalg.norm(q_vec)
    db_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(db_norms, q_norm.T).flatten()

    # Retrieve top 6 most relevant excerpts instead of 3
    top_k = np.argsort(similarities)[::-1][:6]
    context_chunks = []
    for i in top_k:
        src = metadata[i].get('source', 'Clinical Database') if isinstance(metadata[i], dict) else 'Clinical Database'
        context_chunks.append(f"Source: {src}\n{documents[i]}")
    context = "\n\n---\n\n".join(context_chunks)

    full_prompt = f"""Verified Knowledge Base Context:
{context}

Patient Presentation / Case Details:
{prompt}

Provide a structured clinical toxicology consult based on the above information."""

    with st.chat_message("assistant"):
        with st.spinner("Analyzing case and cross-referencing knowledge base..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
                )
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as err:
                st.error(f"API Error Details: {err}")
