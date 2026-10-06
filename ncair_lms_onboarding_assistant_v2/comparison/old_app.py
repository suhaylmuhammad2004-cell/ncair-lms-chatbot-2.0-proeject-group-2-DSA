import os
import json
import time
import requests
import subprocess
import pytesseract
from pdf2image import convert_from_path
from IPython.display import display, Javascript

# LangChain Imports
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr


# ==========================================
# PHASE 1: DATA INGESTION & FAISS BUILD
# ==========================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

all_documents = []

# 1. OCR Processing for PDF Manuals in data/
pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
if not pdf_files:
    print("⚠️ No PDF files found in data/ directory! Skipping PDF OCR processing.")
else:
    pdf_path = pdf_files[0]
    print(f"📄 Processing '{pdf_path}' with OCR...")
    images = convert_from_path(pdf_path)
    for page_num, img in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(img)
        if page_text.strip():
            doc = Document(
                page_content=page_text,
                metadata={"source": os.path.basename(pdf_path), "page": page_num}
            )
            all_documents.append(doc)
    print(f"✅ OCR extraction complete for PDF ({len(images)} pages processed).")

# 2. Ingest Text Knowledge Base Files in data/
txt_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
if not txt_files:
    print("⚠️ No .txt files found in data/ directory!")
else:
    for txt_file in txt_files:
        print(f"📝 Ingesting text knowledge base file: '{txt_file}'...")
        loader = TextLoader(txt_file)
        all_documents.extend(loader.load())

# 3. Build FAISS Vector Database
if all_documents:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""]
    )
    documents_chunks = text_splitter.split_documents(all_documents)
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    faiss_vector_db = FAISS.from_documents(documents_chunks, embedding_model)
else:
    faiss_vector_db = None

def search_ncair_knowledge_base(query: str, top_k: int = 2) -> str:
    """Searches the FAISS vector index for top-k document passages relevant to the user query."""
    if not faiss_vector_db:
        return "No knowledge base documents currently indexed."
    docs = faiss_vector_db.similarity_search(query, k=top_k)
    retrieved_passages = []
    for doc in docs:
        source = doc.metadata.get('source', 'Knowledge Base')
        page = doc.metadata.get('page')
        header = f"[{source} - Page {page}]" if page else f"[{source}]"
        retrieved_passages.append(f"{header}\n{doc.page_content.strip()}")
    return "\n---\n".join(retrieved_passages)


# ==========================================
# PHASE 2: AGENT TOOLS & REGISTRY
# ==========================================
PORTAL_URL_REGISTRY = {
    "main": "https://lms.ncair.nitda.gov.ng",
    "login": "https://lms.ncair.nitda.gov.ng/intern/signin",
    "signin": "https://lms.ncair.nitda.gov.ng/intern/signin",
    "ncair_home": "https://ncair.nitda.gov.ng/",
    "register": "https://lms.ncair.nitda.gov.ng",
    "profile": "https://lms.ncair.nitda.gov.ng/intern/profile",
    "courses": "https://lms.ncair.nitda.gov.ng/intern/courses",
    "track_selection": "https://lms.ncair.nitda.gov.ng/intern/courses",
    "support": "https://ncair.nitda.gov.ng/contact/"
}

def get_portal_link(action: str) -> str:
    action_key = action.lower().strip()
    if action_key in PORTAL_URL_REGISTRY:
        url = PORTAL_URL_REGISTRY[action_key]
        clean_title = action_key.replace('_', ' ').title()
        return f"🔗 [Click here to open the {clean_title} Page]({url})"
    return f"🔗 [Click here to visit the NCAIR LMS Landing Page]({PORTAL_URL_REGISTRY['main']})"

def get_step_guidance(step_number: int) -> str:
    guidance_map = {
        1: "📍 **Step 1 (Profile Setup):** Navigate to 'Edit Profile', complete all required bio fields, and save changes.",
        2: "📍 **Step 2 (ID Verification):** Go to 'Documents', upload a valid ID (National ID/Student ID/Passport) under 2MB, and submit.",
        3: "📍 **Step 3 (Track Selection):** Navigate to 'Track Selection' and choose your assigned department (AI, Robotics, Embedded Systems, Data Science).",
        4: "📍 **Step 4 (Final Submission):** Verify all details on your summary dashboard and click 'Submit Final Registration'."
    }
    return guidance_map.get(step_number, "⚠️ Invalid step number. Please specify a step between 1 and 4.")


# ==========================================
# PHASE 3: ORCHESTRATOR & LLM DISPATCH
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_local_llama(prompt: str, model_name: str = "llama3.2:3b") -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"⚠️ Error from local LLM engine (Status Code: {response.status_code})"
    except Exception as e:
        return f"⚠️ Could not connect to local Ollama server: {str(e)}"

def agentic_rag_orchestrator(user_query: str) -> str:
    query_lower = user_query.lower()

    # Intent 1: Navigation Link Trigger
    if any(keyword in query_lower for keyword in ["link", "url", "where to", "open", "page", "access", "portal"]):
        if "register" in query_lower or "sign up" in query_lower:
            return f"To begin registration, open the portal page here:\n\n{get_portal_link('register')}"
        elif "log" in query_lower or "sign in" in query_lower:
            return f"To log in to your account, use this direct link:\n\n{get_portal_link('login')}"
        elif "profile" in query_lower:
            return f"You can edit your bio and profile details here:\n\n{get_portal_link('profile')}"
        elif "track" in query_lower:
            return f"Access track selection using this page:\n\n{get_portal_link('track_selection')}"
        elif "support" in query_lower or "help" in query_lower:
            return f"You can reach support directly here:\n\n{get_portal_link('support')}"
        else:
            return f"Here is the link to the main NCAIR LMS landing page:\n\n{get_portal_link('main')}"

    # Intent 2: Step Troubleshooting Trigger
    if "stuck" in query_lower or "step" in query_lower:
        if "1" in query_lower or "profile" in query_lower:
            return get_step_guidance(1)
        elif "2" in query_lower or "upload" in query_lower or "id" in query_lower:
            return get_step_guidance(2)
        elif "3" in query_lower or "track" in query_lower:
            return get_step_guidance(3)
        elif "4" in query_lower or "submit" in query_lower:
            return get_step_guidance(4)
        else:
            return "Here is the complete step-by-step onboarding sequence:\n\n" + "\n\n".join([get_step_guidance(i) for i in range(1, 5)])

    # Intent 3: Knowledge Base Search Fallback (FAISS + LLM)
    retrieved_context = search_ncair_knowledge_base(user_query, top_k=2)

    system_prompt = f"""
    You are the official NCAIR LMS Onboarding Assistant.
    Answer the intern's question concisely using ONLY the official context provided below.
    If the provided context does not contain enough details to answer, state clearly that they should contact support at support@ncair.nitda.gov.ng.

    --- OFFICIAL NCAIR CONTEXT ---
    {retrieved_context}
    ------------------------------

    User Question: {user_query}
    Answer:
    """
    return query_local_llama(system_prompt)


# ==========================================
# PHASE 4: GRADIO USER INTERFACE
# ==========================================
def chat_response(message: str, history) -> str:
    return agentic_rag_orchestrator(message)

demo = gr.ChatInterface(
    fn=chat_response,
    title="🇳🇬 NCAIR LMS Intelligent Assistant",
    description="Official Virtual Onboarding & Navigation Guide for Interns & NYSC Members",
    textbox=gr.Textbox(
        placeholder="Ask about registration steps, attendance policies, logbooks, or portal URLs..."
    ),
    chatbot=gr.Chatbot(
        height=480,
        avatar_images=(
            "https://api.iconify.design/lucide:user.svg?color=%23475569",
            "https://api.iconify.design/lucide:bot.svg?color=%2300796b"
        )
    ),
    examples=[
        "Where is the registration link for the portal?",
        "What happens if I pass my tests but fail to meet 75% attendance?",
        "How often do SIWES interns need to sign their logbook?",
        "I am stuck on step 2 of registration",
        "What courses do SIWES interns take in their first cohort?"
    ]
)

if __name__ == "__main__":
    print("🚀 Launching NCAIR Assistant Interface...")
    app, local_url, share_url = demo.launch(
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        share=True,
        prevent_thread_lock=True
    )

    # Auto-open browser tab via JavaScript
    target_url = share_url if share_url else local_url
    if target_url:
        print(f"🔗 Opening interface in new tab: {target_url}")
        display(Javascript(f'window.open("{target_url}", "_blank");'))