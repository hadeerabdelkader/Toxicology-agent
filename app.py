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

# 1. Load Knowledge Vector DB into memory (cached so it loads only once)
@st.cache_resource
def load_resources():
    store_path = "./knowledge_vector_db"
    with open(f"{store_path}/store.pkl", "rb") as f:
        data = pickle.load(f)
    embeddings = np.load(f"{store_path}/embeddings.npy")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embeddings, data["documents"], data["metadata"], embed_model

embeddings, documents, metadata, embed_model = load_resources()

# 2. Configure Gemini Client via Streamlit Secrets or Environment Variable
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
You are a senior Consultant Medical Toxicologist triage officer.

Evaluate cases using this structured framework:
1. Resuscitation & Immediate Danger Signs (Airway, vitals, toxic syndromes).
2. Toxicokinetic Assessment (Weight-based dose in mg/kg vs threshold, absorption window).
3. Targeted Management & Antidote Protocol (Decontamination window, specific antidote, labs, observation timeframe).

GROUNDING RULES:
- Base all guidance strictly on the retrieved knowledge base extracts.
- If the queried substance/protocol is NOT present in the retrieved passages, output:
  "⚠️ Knowledge Base Notice: The queried substance/protocol is not documented in the verified knowledge base. Would you like me to draw from external clinical toxicology references to answer this case?"
- Only provide general external references if explicitly authorized by the user.
"""

# 3. Maintain chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle user clinical input
if prompt := st.chat_input("Enter clinical presentation, patient weight, substance, and ingested dose..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fast in-memory vector similarity lookup
    q_vec = embed_model.encode([prompt], convert_to_numpy=True)
    q_norm = q_vec / np.linalg.norm(q_vec)
    db_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(db_norms, q_norm.T).flatten()

    top_k = np.argsort(similarities)[::-1][:3]
    context = "\n\n---\n\n".join([f"Source: {metadata[i]['source']}\n{documents[i]}" for i in top_k])

    full_prompt = f"Knowledge Base Excerpts:\n{context}\n\nCase Details:\n{prompt}"

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
                st.error (f"API Error Details: {err}")
