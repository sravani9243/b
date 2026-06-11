import streamlit as st
import tempfile
import os
import io
from datetime import datetime

# Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text

# PDF Export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# =========================================================
# INITIALIZATION & UI
# =========================================================
st.set_page_config(page_title="Legal AI Intel", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Document Analysis and Q&A using RAG Framework")
def apply_custom_ui():
    st.markdown("""
        <style>
        .legal-card {
            background: rgba(128, 128, 128, 0.08);
            padding: 24px;
            border-radius: 12px;
            border-left: 5px solid #1e3a8a;
            margin-bottom: 20px;
            color: inherit;
            line-height: 1.6;
        }
        .stButton>button {
            width: 100%; border-radius: 8px; height: 3.2em;
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: white; font-weight: 600;
        }
        .role-badge {
            padding: 8px; border-radius: 6px; text-align: center;
            font-weight: bold; margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize Session States
if "report_data" not in st.session_state: st.session_state.report_data = []
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "current_verdict" not in st.session_state: st.session_state.current_verdict = None
if "verdict_pdf_buffer" not in st.session_state: st.session_state.verdict_pdf_buffer = None

# =========================================================
# TOOLS & ENGINES
# =========================================================
@st.cache_resource
def load_system():
    # Embedding model
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Replace with your actual key or use st.secrets for security
    GROQ_API_KEY = "" 
    
    llm = ChatGroq(
        groq_api_key="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", 
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    return embeds, llm

embeddings, llm = load_system()

def build_legal_rag(files):
    all_chunks = []
    for f in files:
        # Important: use getvalue() to read the file contents correctly in Streamlit
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        all_chunks.extend(loader.load())
        os.remove(tmp_path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(all_chunks)
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    system_prompt = (
        "You are a legal expert. Use the following context to answer the user's question accurately. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

def generate_pdf_report(title, content_list):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1e3a8a"))
    body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=11, leading=14)
    
    elements = [Paragraph(title, title_style), Spacer(1, 12)]
    
    for item in content_list:
        elements.append(Paragraph(f"<b>Module: {item['Module']}</b>", body_style))
        elements.append(Paragraph(f"<i>Input:</i> {item['Input']}", body_style))
        elements.append(Spacer(1, 6))
        
        # Clean output for ReportLab (backslash fix handled here by variables)
        clean_output = item['Output'].replace('\n', '<br/>')
        elements.append(Paragraph(clean_output, body_style))
        
        elements.append(Spacer(1, 15))
        elements.append(HRFlowable(width="100%", color=colors.grey))
        elements.append(Spacer(1, 10))
        
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================================================
# MAIN APP UI
# =========================================================
apply_custom_ui()
st.sidebar.header("⚖️ User Access")
user_role = st.sidebar.selectbox("Select Role", ["Public", "Lawyer", "Judge"])

role_bg = "#FEE2E2" if user_role == "Judge" else "#DBEAFE"
st.sidebar.markdown(f'<div class="role-badge" style="background:{role_bg}; color:black;">Mode: {user_role}</div>', unsafe_allow_html=True)

nav_options = ["Legal Assistant (RAG)", "Contract Comparison", "Export Center"]
if user_role == "Judge": nav_options.insert(1, "Judgment Mode")

nav = st.sidebar.radio("Navigation", nav_options)
uploads = st.sidebar.file_uploader("📂 Upload Case PDFs", type="pdf", accept_multiple_files=True)

# Trigger RAG building if files are uploaded
if uploads and "rag_chain" not in st.session_state:
    with st.status("Reading Legal Documents...") as status:
        st.session_state.rag_chain = build_legal_rag(uploads)
        status.update(label="Index Complete!", state="complete")

# --- 1. LEGAL ASSISTANT ---
if nav == "Legal Assistant (RAG)":
    st.header("💬 AI Legal Assistant")
    input_col, mic_col = st.columns([0.85, 0.15])
    
    with mic_col:
        st.write(" ")
        voice_result = speech_to_text(language='en', start_prompt="🎤", stop_prompt="🛑", just_once=True, key='rag_mic')
        if voice_result:
            st.session_state.transcript = voice_result
            st.rerun()

    with input_col:
        query = st.text_input("Query", value=st.session_state.transcript, placeholder="Ask something about the uploaded files...", label_visibility="collapsed")
    
    if st.button("⚖️ Run Analysis") and query:
        if "rag_chain" in st.session_state:
            with st.spinner("Searching case law..."):
                res = st.session_state.rag_chain.invoke({"input": query})
                answer = res["answer"]
                
                # Format for display (avoiding backslashes in f-string)
                display_answer = answer.replace('\n', '<br/>')
                st.markdown(f'<div class="legal-card">{display_answer}</div>', unsafe_allow_html=True)
                
                st.session_state.report_data.append({"Module": "RAG Assistant", "Input": query, "Output": answer})
        else:
            st.warning("Please upload PDF documents in the sidebar first.")

# --- 2. JUDGMENT MODE (JUDGE ONLY) ---
elif nav == "Judgment Mode" and user_role == "Judge":
    st.header("👨‍⚖️ Judgment Suite")
    
    c_ref = st.text_input("Case Reference Number")
    f_sum = st.text_area("Summary of Facts & Evidence", height=200)
    
    if st.button("⚖️ Draft Judgment") and f_sum:
        with st.spinner("AI Deliberating based on facts..."):
            prompt = f"Draft a formal, high-authority legal judgment for Case {c_ref}. Use professional legal tone. Facts: {f_sum}"
            res = llm.invoke(prompt)
            st.session_state.current_verdict = res.content
            
            # Generate and store the buffer
            pdf_buf = generate_pdf_report(
                f"Official Judgment: {c_ref}", 
                [{"Module": "Judgment", "Input": c_ref, "Output": res.content}]
            )
            st.session_state.verdict_pdf_buffer = pdf_buf.getvalue()
            st.session_state.report_data.append({"Module": "Judgment", "Input": c_ref, "Output": res.content})

    if st.session_state.current_verdict:
        # Fixed: Move replacement outside the f-string to avoid SyntaxError
        verdict_display = st.session_state.current_verdict.replace("\n", "<br/>")
        st.markdown(f'<div class="legal-card">{verdict_display}</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download This Judgment PDF",
            data=st.session_state.verdict_pdf_buffer,
            file_name=f"Judgment_{c_ref}.pdf",
            mime="application/pdf"
        )

# --- 3. CONTRACT COMPARISON ---
elif nav == "Contract Comparison":
    st.header("🔍 Contractual Clause Comparison")
    c1, c2 = st.columns(2)
    doc_a = c1.file_uploader("Upload Contract A", type="pdf", key="ca")
    doc_b = c2.file_uploader("Upload Contract B", type="pdf", key="cb")
    
    if doc_a and doc_b and st.button("⚡ Compare Documents"):
        def extract_text(f):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                t.write(f.getvalue())
                t_path = t.name
            loader = PyPDFLoader(t_path)
            txt = " ".join([d.page_content for d in loader.load()])
            os.remove(t_path)
            return txt[:5000] # Safe slice for context
        
        with st.spinner("Cross-referencing clauses..."):
            text_a = extract_text(doc_a)
            text_b = extract_text(doc_b)
            
            comparison_prompt = f"""Compare these two contracts. 
            Focus on: Liability, Termination, and Payment.
            Contract A: {text_a}
            ---
            Contract B: {text_b}"""
            
            diff = llm.invoke(comparison_prompt).content
            
            # Fixed: Move replacement outside the f-string
            diff_display = diff.replace("\n", "<br/>")
            st.markdown(f'<div class="legal-card">{diff_display}</div>', unsafe_allow_html=True)
            st.session_state.report_data.append({"Module": "Comparison", "Input": "Contract A vs B", "Output": diff})

# --- 4. EXPORT CENTER ---
elif nav == "Export Center":
    st.header("📋 Final Case Export")
    
    if st.session_state.report_data:
        st.write(f"You have **{len(st.session_state.report_data)}** items in your current session report.")
        
        full_report_pdf = generate_pdf_report(
            f"Legal Intelligence Summary - {datetime.now().strftime('%Y-%m-%d')}", 
            st.session_state.report_data
        )
        
        st.download_button(
            label="📥 Download Comprehensive Activity Report",
            data=full_report_pdf,
            file_name=f"Legal_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
        
        if st.button("🗑️ Clear Session Data"):
            st.session_state.report_data = []
            st.session_state.current_verdict = None
            st.session_state.verdict_pdf_buffer = None
            st.rerun()
    else:
        st.info("Your report is currently empty. Run an analysis or draft a judgment to see data here.")