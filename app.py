import streamlit as st
import os
import nltk
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return sbert, cross, classifier

model, cross_model, classifier = load_models()

# --- UI Setup ---
st.set_page_config(page_title="AI Paper Evaluator Pro", layout="wide")
st.title("AI vs Human: Comparative Paper Evaluation")

# Tabs for Manual vs Automated Testing
tab1, tab2 = st.tabs(["Individual Evaluation", "Automated Bulk Testing"])

# --- Sidebar Global Settings ---
st.sidebar.header("1. Global Settings")
max_score = st.sidebar.number_input("Set Maximum Score", min_value=1, value=5) # Changed default to 5 for your dataset

# Initialize session state for history
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
    labels = ["Natural Language Processing", "Cloud Computing", "Machine Learning", "Computer Networks", "Databases", "Software Engineering", "Education",# Technology & Engineering
    "Machine Learning & AI", "Full-Stack Development", "Cloud Computing & DevOps", 
    "Cybersecurity", "Data Science", "Mobile App Development",
    
    # Science & Nature
    "Space Exploration & Astronomy", "Environmental Science", "Physics & Chemistry",
    "Biology & Medicine", "Climate Change",
    
    # Humanities & Society
    "World History", "Political Science", "Philosophy & Ethics", 
    "Economics & Finance", "Geography",
    
    # Education & Growth
    "Pedagogy & Teaching Methods", "Language Learning", "Early Childhood Education",
    
    # Lifestyle & Sports
    "Professional Sports & Athletics", "Health & Fitness", "Art & Literature",
              "Music & Entertainment", "Travel & Culture"]
    result = classifier(text[:1000], labels)
    return result['labels'][0]

def calculate_full_evaluation(base_text, student_text, max_val, h_score):
    # 1. Semantic Analysis
    emb = model.encode([base_text, student_text])
    sbert_sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    cross_sim = max(min(cross_model.predict([(base_text, student_text)])[0], 1), 0)
    
    # 2. Topic Detection
    ref_topic = detect_topic(base_text)
    stu_topic = detect_topic(student_text)
    topic_match = 1 if ref_topic == stu_topic else 0
    
    # 3. Scoring Logic
    coverage = keyword_coverage(base_text, student_text)
    relevance = (0.4 * sbert_sim + 0.4 * cross_sim + 0.2 * coverage)
    if topic_match == 0: relevance *= 0.4
    
    features = get_feature_vector(student_text)
    quality_weight = (min(features['word_count'] / 10, 20) / 20) * max_val
    
    score = (0.6 * (max_val * relevance) + 0.4 * quality_weight)
    
    # Final Adjustments
    if topic_match == 0 and relevance < 0.5: score = min(score, 3)
    length_factor = min(features['word_count'] / 80, 1.0)
    score *= (0.85 + 0.15 * length_factor)
    score = min(round(score, 2), max_val)
    
    return {
        "score": score,
        "ref_topic": ref_topic,
        "stu_topic": stu_topic,
        "topic_match": "Yes" if topic_match else "No",
        "features": features,
        "deviation": round(abs(score - h_score), 2)
    }

# ===============================
# 📥 TAB 1: INDIVIDUAL EVALUATION
# ===============================
with tab1:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Human Evaluation")
    human_score = st.sidebar.slider("Assign Human Score", 0.0, float(max_score), float(max_score/2))

    col_ref, col_stud = st.columns(2)
    with col_ref:
        st.subheader("Reference Answer")
        base_mode = st.radio("Base Input Method", ["Text", "Upload Image/PDF"], key="base_mode")
        base_text = ""
        if base_mode == "Text":
            base_text = st.text_area("Paste reference answer...", height=200, key="txt_ref")
        else:
            base_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "pdf"], key="up_ref")
            if base_file: base_text = extract_text(base_file)

    with col_stud:
        st.subheader("Student Answer")
        stud_mode = st.radio("Student Input Method", ["Type/Paste Text", "Upload Image/PDF"], key="stud_mode")
        student_text = ""
        if stud_mode == "Type/Paste Text":
            student_text = st.text_area("Paste student answer...", height=200, key="txt_stud")
        else:
            stud_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "pdf"], key="up_stud")
            if stud_file: student_text = extract_text(stud_file)

    if st.button("Run Comparative Analysis"):
        if not base_text.strip() or not student_text.strip():
            st.error("Please provide both texts.")
        else:
            with st.spinner("Analyzing..."):
                res = calculate_full_evaluation(base_text, student_text, max_score, human_score)
                
                # Add to history
                st.session_state.history.append({
                    "Trial": len(st.session_state.history) + 1,
                    "Machine Score": res['score'],
                    "Human Score": human_score,
                    "Deviation": res['deviation'],
                    "Topic Match": res['topic_match']
                })

                # Display Metrics
                st.markdown("---")
                st.header("Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Machine Score", res['score'])
                m2.metric("Human Score", human_score)
                m3.metric("Deviation", res['deviation'], delta=f"-{res['deviation']}" if res['deviation'] > 0 else "0", delta_color="inverse")
                m4.metric("Topic Match", res['topic_match'])

                # Display Topics
                st.markdown(f"**Reference Topic:** <span style='color:#1f77b4'>{res['ref_topic']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Student Topic:** <span style='color:#ff7f0e'>{res['stu_topic']}</span>", unsafe_allow_html=True)

    # Visualization (History & Trend)
    if len(st.session_state.history) > 0:
        st.markdown("---")
        df_hist = pd.DataFrame(st.session_state.history)
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("Benchmark History")
            st.dataframe(df_hist, use_container_width=True)
        with c2:
            st.subheader("Performance Trend")
            if len(df_hist) >= 3:
                fig, ax = plt.subplots()
                ax.plot(df_hist['Trial'], df_hist['Machine Score'], label='Machine', marker='o')
                ax.plot(df_hist['Trial'], df_hist['Human Score'], label='Human', marker='s', linestyle='--')
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Trend appears after 3 trials.")

# ===============================
# 📥 TAB 2: AUTOMATED TESTING (ASAP 2.0)
# ===============================
with tab2:
    st.header("ASAP 2.0 Batch Evaluator")
    uploaded_test = st.file_uploader("Upload Kaggle Sample (CSV)", type="csv", key="kaggle_up")
    test_ref = st.text_area("Reference Answer for Batch Test", height=150, key="batch_ref")
    
    if st.button("Start Batch Evaluation") and uploaded_test and test_ref:
        test_df = pd.read_csv(uploaded_test)
        
        # KEY FIX: Mapping columns to what's actually in your CSV
        # If 'full_text' and 'score' exist, use them. Otherwise fallback.
        text_col = 'full_text' if 'full_text' in test_df.columns else 'essay'
        score_col = 'score' if 'score' in test_df.columns else 'domain1_score'
        
        if text_col not in test_df.columns or score_col not in test_df.columns:
            st.error(f"Columns not found! CSV needs '{text_col}' and '{score_col}'. Found: {list(test_df.columns)}")
        else:
            batch_results = []
            prog = st.progress(0)
            
            for idx, row in test_df.iterrows():
                actual_h = row[score_col]
                eval_res = calculate_full_evaluation(test_ref, row[text_col], max_score, actual_h)
                
                batch_results.append({
                    "Human Score": actual_h,
                    "AI Score": eval_res['score'],
                    "Error": eval_res['deviation'],
                    "Topic": eval_res['stu_topic']
                })
                prog.progress((idx + 1) / len(test_df))
            
            res_df = pd.DataFrame(batch_results)
            st.write("### Accuracy Report")
            st.dataframe(res_df)
            st.metric("Mean Absolute Error (MAE)", round(res_df['Error'].mean(), 2))
