# =============================================
# AI Legal Contract Analyzer (FINAL FIXED VERSION)
# =============================================

import streamlit as st
import fitz
import pdfplumber
import nltk
import re
import pandas as pd
import plotly.express as px

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

nltk.download('punkt', quiet=True)

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="AI Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            background-color: #1a1a2e;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
        }
        .risk-high { color: #e63946; font-weight: bold; }
        .risk-medium { color: #f4a261; font-weight: bold; }
        .risk-low { color: #2a9d8f; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# =============================================
# TEXT EXTRACTION
# =============================================
def extract_text(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    return text

# =============================================
# CLAUSE SEGMENTATION
# =============================================
def split_into_clauses(text):
    pattern = r"(?:\n\d+\.\s|\n[A-Z ]{3,}\n|Section\s\d+|\n- )"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if len(p.strip()) > 50]

# =============================================
# LOAD MODELS
# FIX: uses text2text-generation + flan-t5-small
#      works on ALL transformers versions including 5.3.0
# =============================================
@st.cache_resource
def load_models():
    summarizer = pipeline(
        "text2text-generation",          # ✅ NOT "summarization"
        model="google/flan-t5-small"     # ✅ NOT "facebook/bart-large-cnn"
    )
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, similarity_model  # ✅ exactly 2 values

# =============================================
# SUMMARIZATION
# FIX: uses generated_text key, not summary_text
# =============================================
def summarize(text, summarizer):
    text = text[:2000]
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    summaries = []
    for chunk in chunks[:4]:
        if len(chunk.strip()) < 30:
            continue
        try:
            result = summarizer(
                f"summarize: {chunk}",   # ✅ flan-t5 needs this prefix
                max_new_tokens=100,
                do_sample=False
            )
            summaries.append(result[0]['generated_text'])  # ✅ NOT 'summary_text'
        except Exception as e:
            summaries.append(f"[Could not summarize: {e}]")
    return " ".join(summaries) if summaries else "No summary available."

# =============================================
# ML-BASED RISK CLASSIFIER
# =============================================
@st.cache_resource
def train_risk_model():
    clauses = [
        "The party shall indemnify and be liable for all damages and penalties",
        "Confidential information must not be disclosed to any third party",
        "This agreement may be terminated immediately under breach of contract",
        "Standard payment terms apply within 30 days of invoice",
        "The warranty period is limited to 90 days from purchase",
        "Governing law shall be the jurisdiction of the state of New York",
        "Either party may terminate with 30 days written notice",
        "The licensee shall not sublicense or transfer any rights",
        "Force majeure events shall suspend obligations under this contract",
        "Intellectual property rights remain with the original owner",
    ]
    labels = ["HIGH", "MEDIUM", "HIGH", "LOW", "MEDIUM",
              "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clauses)
    model = LogisticRegression(max_iter=200)
    model.fit(X, labels)
    return model, vectorizer

def predict_risk(clauses, model, vectorizer):
    if not clauses:
        return []
    X = vectorizer.transform(clauses)
    return model.predict(X)

# =============================================
# SEMANTIC SIMILARITY
# =============================================
def compute_similarity(clauses1, clauses2, model):
    if not clauses1 or not clauses2:
        return None
    emb1 = model.encode(clauses1, convert_to_tensor=True)
    emb2 = model.encode(clauses2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2)

# =============================================
# REPORT GENERATION
# =============================================
def create_report(summary1, summary2, risks1, risks2, sim_score, filename="report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "AI Legal Contract Analysis Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, "Generated by AI Legal Contract Analyzer")

    y = height - 120

    def write_section(title, content):
        nonlocal y
        if y < 100:
            c.showPage()
            y = height - 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, title)
        y -= 20
        c.setFont("Helvetica", 9)
        for line in content.split(". "):
            line = line.strip()
            if not line:
                continue
            while len(line) > 90:
                c.drawString(60, y, line[:90])
                line = line[90:]
                y -= 14
                if y < 100:
                    c.showPage()
                    y = height - 60
            c.drawString(60, y, line + ".")
            y -= 14
            if y < 100:
                c.showPage()
                y = height - 60
        y -= 10

    write_section("Contract A — Summary:", summary1)
    write_section("Contract B — Summary:", summary2)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Risk Analysis:")
    y -= 20
    c.setFont("Helvetica", 10)
    for label, risks in [("Contract A", risks1), ("Contract B", risks2)]:
        risk_list = list(risks)
        c.drawString(60, y,
            f"{label} — HIGH: {risk_list.count('HIGH')}  "
            f"MEDIUM: {risk_list.count('MEDIUM')}  "
            f"LOW: {risk_list.count('LOW')}")
        y -= 16

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Semantic Similarity Score: {sim_score:.3f}")
    c.save()

# =============================================
# UI
# =============================================
st.title("⚖️ AI Legal Contract Analyzer (ML Powered)")
st.markdown("Upload two legal contracts (PDF) to compare, summarize, and detect risk clauses.")

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("📄 Upload Contract A", type=["pdf"])
with col2:
    file2 = st.file_uploader("📄 Upload Contract B", type=["pdf"])

if file1 and file2:
    st.success("✅ Files uploaded successfully!")

    with st.spinner("Extracting text from PDFs..."):
        text1 = extract_text(file1)
        text2 = extract_text(file2)

    with st.spinner("Segmenting clauses..."):
        clauses1 = split_into_clauses(text1)
        clauses2 = split_into_clauses(text2)

    col1, col2 = st.columns(2)
    col1.metric("Clauses in Contract A", len(clauses1))
    col2.metric("Clauses in Contract B", len(clauses2))

    # ✅ Unpack exactly 2 values
    with st.spinner("⏳ Loading AI models — first run downloads ~300MB, please wait..."):
        summarizer, similarity_model = load_models()
        risk_model, vectorizer = train_risk_model()

    # ✅ Pass exactly 2 args
    with st.spinner("Generating summaries..."):
        summary1 = summarize(text1, summarizer)
        summary2 = summarize(text2, summarizer)

    st.subheader("📝 Summaries")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contract A Summary**")
        st.info(summary1)
    with col2:
        st.markdown("**Contract B Summary**")
        st.info(summary2)

    with st.spinner("Running risk classification..."):
        risks1 = predict_risk(clauses1, risk_model, vectorizer)
        risks2 = predict_risk(clauses2, risk_model, vectorizer)

    st.subheader("📊 Risk Distribution")
    df = pd.DataFrame({
        "Risk Level": ["HIGH", "MEDIUM", "LOW"],
        "Contract A": [
            list(risks1).count("HIGH"),
            list(risks1).count("MEDIUM"),
            list(risks1).count("LOW")
        ],
        "Contract B": [
            list(risks2).count("HIGH"),
            list(risks2).count("MEDIUM"),
            list(risks2).count("LOW")
        ]
    })

    fig = px.bar(
        df,
        x="Risk Level",
        y=["Contract A", "Contract B"],
        barmode="group",
        color_discrete_sequence=["#e63946", "#457b9d"],
        title="Risk Level Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 Top HIGH Risk Clauses")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Contract A**")
        high_risk_1 = [c for c, r in zip(clauses1, risks1) if r == "HIGH"]
        if high_risk_1:
            for clause in high_risk_1[:5]:
                st.error(f"• {clause[:200]}...")
        else:
            st.success("No HIGH risk clauses found.")

    with col2:
        st.markdown("**Contract B**")
        high_risk_2 = [c for c, r in zip(clauses2, risks2) if r == "HIGH"]
        if high_risk_2:
            for clause in high_risk_2[:5]:
                st.error(f"• {clause[:200]}...")
        else:
            st.success("No HIGH risk clauses found.")

    with st.spinner("Computing semantic similarity..."):
        sim = compute_similarity(clauses1[:20], clauses2[:20], similarity_model)

    sim_score = sim.mean().item() if sim is not None else 0.0

    st.subheader("🔗 Semantic Similarity Score")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="Cosine Similarity (0 = different, 1 = identical)",
            value=f"{sim_score:.3f}",
            delta="High similarity" if sim_score > 0.7
                  else ("Moderate" if sim_score > 0.4 else "Low similarity")
        )

    st.subheader("📥 Download Report")
    if st.button("Generate & Download PDF Report"):
        with st.spinner("Creating report..."):
            create_report(summary1, summary2, risks1, risks2, sim_score)
        with open("report.pdf", "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name="contract_analysis_report.pdf",
                mime="application/pdf"
            )

elif file1 or file2:
    st.warning("⚠️ Please upload **both** contracts to begin analysis.")

else:
    st.markdown("""
    ### How to use:
    1. Upload **Contract A** (PDF)
    2. Upload **Contract B** (PDF)
    3. The system will automatically:
       - Extract and segment clauses
       - Summarize each contract
       - Classify risk levels (HIGH / MEDIUM / LOW)
       - Compare contracts by semantic similarity
       - Generate a downloadable PDF report
    """)
