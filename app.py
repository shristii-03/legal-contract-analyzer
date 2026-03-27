import streamlit as st
import fitz
import pdfplumber
import nltk
import re
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
st.set_page_config(page_title="AI Legal Contract Analyzer", page_icon="⚖️", layout="wide")
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
def split_into_clauses(text):
    pattern = r"(?:\n\d+\.\s|\n[A-Z ]{3,}\n|Section\s\d+|\n- )"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if len(p.strip()) > 50]
def summarize(text):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        s = LsaSummarizer()
        result = " ".join(str(x) for x in s(parser.document, 6))
        return result if result.strip() else text[:500]
    except:
        return text[:500]
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def train_risk_model():
    clauses = ["The party shall indemnify and be liable for all damages","Confidential information must not be disclosed","This agreement may be terminated under breach","Standard payment terms apply","The warranty period is limited","Either party may terminate with notice","The licensee shall not sublicense rights","Force majeure events suspend obligations","Intellectual property rights remain with owner","Penalty clause applies for non-performance"]
    labels = ["HIGH","MEDIUM","HIGH","LOW","MEDIUM","LOW","MEDIUM","MEDIUM","HIGH","HIGH"]
    v = TfidfVectorizer()
    X = v.fit_transform(clauses)
    m = LogisticRegression(max_iter=500)
    m.fit(X, labels)
    return m, v
def predict_risk(clauses, model, vectorizer):
    if not clauses:
        return []
    return model.predict(vectorizer.transform(clauses))
def compute_similarity(c1, c2, model):
    if not c1 or not c2:
        return None
    return util.cos_sim(model.encode(c1, convert_to_tensor=True), model.encode(c2, convert_to_tensor=True))
def create_report(s1, s2, r1, r2, score):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    c = canvas.Canvas("report.pdf", pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h-60, "AI Legal Contract Analysis Report")
    y = h-100
    for title, text in [("Contract A Summary:", s1), ("Contract B Summary:", s2)]:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, title)
        y -= 20
        c.setFont("Helvetica", 9)
        for chunk in [text[i:i+90] for i in range(0, min(len(text),900), 90)]:
            c.drawString(60, y, chunk)
            y -= 14
            if y < 80:
                c.showPage()
                y = h-60
        y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Risk Summary:")
    y -= 18
    c.setFont("Helvetica", 10)
    for label, risks in [("Contract A", list(r1)), ("Contract B", list(r2))]:
        c.drawString(60, y, f"{label} — HIGH: {risks.count('HIGH')}  MEDIUM: {risks.count('MEDIUM')}  LOW: {risks.count('LOW')}")
        y -= 16
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y-10, f"Similarity Score: {score:.3f}")
    c.save()
st.title("⚖️ AI Legal Contract Analyzer")
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload Contract A", type=["pdf"])
with col2:
    file2 = st.file_uploader("Upload Contract B", type=["pdf"])
if file1 and file2:
    st.success("Files uploaded!")
    text1 = extract_text(file1)
    text2 = extract_text(file2)
    clauses1 = split_into_clauses(text1)
    clauses2 = split_into_clauses(text2)
    c1, c2 = st.columns(2)
    c1.metric("Clauses A", len(clauses1))
    c2.metric("Clauses B", len(clauses2))
    with st.spinner("Loading models..."):
        sim_model = load_similarity_model()
        risk_model, vectorizer = train_risk_model()
    with st.spinner("Summarizing..."):
        s1 = summarize(text1)
        s2 = summarize(text2)
    st.subheader("Summaries")
    c1, c2 = st.columns(2)
    c1.info(s1)
    c2.info(s2)
    risks1 = predict_risk(clauses1, risk_model, vectorizer)
    risks2 = predict_risk(clauses2, risk_model, vectorizer)
    st.subheader("Risk Distribution")
    df = pd.DataFrame({"Risk":["HIGH","MEDIUM","LOW"],"Contract A":[list(risks1).count("HIGH"),list(risks1).count("MEDIUM"),list(risks1).count("LOW")],"Contract B":[list(risks2).count("HIGH"),list(risks2).count("MEDIUM"),list(risks2).count("LOW")]})
    st.plotly_chart(px.bar(df, x="Risk", y=["Contract A","Contract B"], barmode="group", color_discrete_sequence=["#e63946","#457b9d"]), use_container_width=True)
    st.subheader("High Risk Clauses")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Contract A**")
        hr1 = [c for c,r in zip(clauses1,risks1) if r=="HIGH"]
        [st.error(f"• {c[:200]}...") for c in hr1[:5]] if hr1 else st.success("None found.")
    with c2:
        st.markdown("**Contract B**")
        hr2 = [c for c,r in zip(clauses2,risks2) if r=="HIGH"]
        [st.error(f"• {c[:200]}...") for c in hr2[:5]] if hr2 else st.success("None found.")
    sim = compute_similarity(clauses1[:20], clauses2[:20], sim_model)
    score = sim.mean().item() if sim is not None else 0.0
    st.subheader("Similarity Score")
    st.metric("Cosine Similarity", f"{score:.3f}")
    if st.button("Download Report"):
        create_report(s1, s2, risks1, risks2, score)
        with open("report.pdf","rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")
elif file1 or file2:
    st.warning("Please upload both contracts.")
else:
    st.info("Upload two PDF contracts above to begin analysis.")
