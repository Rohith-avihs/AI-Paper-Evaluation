import cv2
import pytesseract
import nltk
import re
import numpy as np
from PIL import Image
import pdfplumber
import tempfile

# --- Ensure NLTK resources ---
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ===============================
# 🔹 TEXT EXTRACTION
# ===============================
def extract_text(file):
    """
    Handles:
    - Uploaded images (Streamlit)
    - Image file paths
    - PDFs
    """

    try:
        # --- If file is uploaded (Streamlit UploadedFile) ---
        if hasattr(file, "type"):

            # ✅ IMAGE FILE
            if "image" in file.type:
                image = Image.open(file)
                image = np.array(image)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                processed = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]

                return pytesseract.image_to_string(processed, config='--psm 6')

            # ✅ PDF FILE
            elif "pdf" in file.type:
                text = ""
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text

        # --- If file is a path ---
        elif isinstance(file, str):
            image = cv2.imread(file)
            if image is None:
                return ""

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            return pytesseract.image_to_string(processed, config='--psm 6')

    except Exception as e:
        print("Extraction Error:", e)
        return ""

    return ""


# ===============================
# 🔹 FEATURE EXTRACTION
# ===============================
def get_feature_vector(text):
    """Extract linguistic features"""

    words = text.split()
    word_count = len(words)

    # --- Sentences ---
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_count = len(sentences)

    # --- Word stats ---
    long_words = [w for w in words if len(re.sub(r'\W+', '', w)) > 5]
    avg_word_len = (
        sum(len(w) for w in words) / word_count if word_count > 0 else 0
    )

    # --- POS tagging ---
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    nouns = len([w for w, t in pos_tags if t.startswith('NN')])
    verbs = len([w for w, t in pos_tags if t.startswith('VB')])
    adj = len([w for w, t in pos_tags if t.startswith('JJ')])

    return {
        'word_count': word_count,
        'sent_count': sent_count,
        'long_word_count': len(long_words),
        'avg_word_len': round(avg_word_len, 2),
        'noun_count': nouns,
        'verb_count': verbs,
        'adj_count': adj
    }
