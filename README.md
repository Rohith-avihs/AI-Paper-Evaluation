# AI vs Human: Comparative Paper Evaluation System

An advanced automated essay scoring (AES) system that leverages Natural Language Processing (NLP) to provide objective grading. This tool compares student submissions against reference keys using multiple AI architectures to ensure fairness and accuracy.

## 🚀 Key Features
* **Dual-Model Analysis:** Uses SBERT for semantic meaning and Cross-Encoders for deep contextual similarity.
* **Comparative Visualization:** Generates real-time Matplotlib graphs comparing "Human Scores" vs "AI Predicted Scores" to identify grading bias.
* **Feature Extraction:** Analyzes linguistic features including word count, sentence structure, and keyword density.
* **Topic Modeling:** Automatically detects the subject matter (NLP, Cloud Computing, etc.) using Zero-Shot Classification.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **NLP Models:** Sentence-Transformers (all-MiniLM-L6-v2), Transformers (BART-Large-MNLI)
* **Data Science:** Pandas, Matplotlib, Scikit-learn
* **Language:** Python 3.x

## 📋 How to Run
1. Clone the repository
2. Create a virtual environment: `python -m venv ml_env`
3. Install dependencies: `pip install -r requirements.txt`
4. Launch the app: `streamlit run app.py`
