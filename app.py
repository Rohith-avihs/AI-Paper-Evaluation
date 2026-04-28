import streamlit as st
import os
import nltk
import logging
import pandas as pd
import matplotlib.pyplot as plt
from processor import extract_text, get_feature_vector
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# --- Silence logs ---
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- NLTK ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

load_nltk()

# --- Models ---
@st.cache_resource
def load_models():
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")
    return sbert, cross, classifier

model, cross_model, classifier = load_models()

# --- UI Setup ---
st.set_page_config(page_title="AI Paper Evaluator Pro", layout="wide")
st.title("AI vs Human: Comparative Paper Evaluation")

# --- Sidebar Evaluation & Human Correction Settings ---
st.sidebar.header("1. Global Settings")
max_score = st.sidebar.number_input("Set Maximum Score", min_value=1, value=10)

st.sidebar.markdown("---")
st.sidebar.header("2. Human Evaluation (Benchmark)")
human_score = st.sidebar.slider("Assign Human Score", 0.0, float(max_score), float(max_score/2), 
                                help="Assign the score a human teacher would give to compare with AI.")

# Initialize session state for the benchmark table
if 'history' not in st.session_state:
    st.session_state.history = []

# ===============================
# 🔧 HELPERS
# ===============================
def keyword_coverage(base, student):
    base_words = set(base.lower().split())
    stud_words = set(student.lower().split())
    important = [w for w in base_words if len(w) > 5]
    if not important: return 0
    matched = [w for w in important if w in stud_words]
    return len(matched) / len(important)

def detect_topic(text):
    labels = ["Natural Language Processing", "Cloud Computing", "Machine Learning", 
              "Computer Networks", "Databases", "Software Engineering"]
    result = classifier(text[:1000], labels)
    return result['labels'][0]

# ===============================
# 📥 INPUT UI
# ===============================
col_ref, col_stud = st.columns(2)

with col_ref:
    st.subheader("Reference Answer")
    base_mode = st.radio("Base Input Method", ["Type/Paste Text", "Upload Image/PDF"], key="base_mode")
    base_text = ""
    if base_mode == "Type/Paste Text":
        base_text = st.text_area("Paste reference answer...", height=200)
    else:
        base_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"], key="base_up")
        if base_file:
            base_text = extract_text(base_file)
            st.success("Text extracted!")

with col_stud:
    st.subheader("Student Answer")
    stud_mode = st.radio("Student Input Method", ["Type/Paste Text", "Upload Image/PDF"], key="stud_mode")
    student_text = ""
    if stud_mode == "Type/Paste Text":
        student_text = st.text_area("Paste student answer...", height=200)
    else:
        stud_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"], key="stud_up")
        if stud_file:
            student_text = extract_text(stud_file)
            st.success("Text extracted!")

# ===============================
# 🚀 EVALUATION & COMPARATIVE ANALYSIS
# ===============================
if st.button("Run Comparative Analysis"):
    if not base_text.strip() or not student_text.strip():
        st.error("Please provide both texts.")
    else:
        with st.spinner("Performing Transformer-based Analysis..."):
            emb = model.encode([base_text, student_text])
            sbert_sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            cross_sim = cross_model.predict([(base_text, student_text)])[0]
            cross_sim = max(min(cross_sim, 1), 0)

            coverage = keyword_coverage(base_text, student_text)
            ref_topic = detect_topic(base_text)
            stu_topic = detect_topic(student_text)
            topic_match = 1 if ref_topic == stu_topic else 0

            relevance = (0.4 * sbert_sim + 0.4 * cross_sim + 0.2 * coverage)
            if topic_match == 0: relevance *= 0.4
            relevance = min(relevance, 1.0)

            features = get_feature_vector(student_text)
            ai_quality = min(features['word_count'] / 10, 20)
            quality_weight = (ai_quality / 20) * max_score

            final_machine_score = (0.6 * (max_score * relevance) + 0.4 * quality_weight)
            if topic_match == 0 and relevance < 0.5:
                final_machine_score = min(final_machine_score, 3)
            
            length_factor = min(features['word_count'] / 80, 1.0)
            final_machine_score *= (0.85 + 0.15 * length_factor)
            final_machine_score = min(round(final_machine_score, 2), max_score)

            deviation = round(abs(final_machine_score - human_score), 2)
            
            st.session_state.history.append({
                "Trial": len(st.session_state.history) + 1,
                "Machine Score": final_machine_score,
                "Human Score": human_score,
                "Deviation": deviation,
                "Topic Match": "Yes" if topic_match else "No"
            })

            st.markdown("---")
            st.header("Comparative Analysis Results")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Machine Score", f"{final_machine_score}")
            m2.metric("Human Score", f"{human_score}")
            m3.metric("Deviation", f"{deviation}", delta=f"-{deviation}" if deviation > 0 else "0", delta_color="inverse")
            m4.metric("Topic Match", "YES" if topic_match else "NO")

            # ✅ --- ADDED TOPIC DISPLAY ---
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold;'> Reference Topic: "
                f"<span style='color:#1f77b4'>{ref_topic}</span></div>",
                unsafe_allow_html=True
            )

            st.markdown(
                    f"<div style='font-size:20px; font-weight:bold;'>Student Topic: "
                f"<span style='color:#ff7f0e'>{stu_topic}</span></div>",
                unsafe_allow_html=True
            )

# ===============================
# 📈 TREND VISUALIZATION
# ===============================
if len(st.session_state.history) > 0:
    st.markdown("---")
    df = pd.DataFrame(st.session_state.history)
    
    col_table, col_chart = st.columns([1, 1.5])
    
    with col_table:
        st.subheader("Benchmark History")
        st.dataframe(df, use_container_width=True)
    
    with col_chart:
        st.subheader("Performance Trend")
        if len(df) >= 3:
            fig, ax = plt.subplots()
            ax.plot(df['Trial'], df['Machine Score'], label='Machine', marker='o', color='#1f77b4')
            ax.plot(df['Trial'], df['Human Score'], label='Human', marker='s', linestyle='--', color='#ff7f0e')
            ax.fill_between(df['Trial'], df['Machine Score'], df['Human Score'], color='gray', alpha=0.2, label='Deviation')
            ax.set_xlabel("Evaluation No.")
            ax.set_ylabel("Score")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("The trend graph will appear after 3 evaluations.")

    if student_text:
        col_text, col_feat = st.columns(2)
        with col_text:
            st.write("**Processed Student Text:**")
            st.markdown(
                f"<div style='font-family:Courier New; font-size:16px'>{student_text}</div>",
                unsafe_allow_html=True
            )
        with col_feat:
            st.write("**Linguistic Features:**")
            st.json(features)

