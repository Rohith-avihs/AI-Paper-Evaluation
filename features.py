import re

def extract_surface_features(text):
    # Clean the text (remove extra whitespaces)
    text = text.strip()
    
    # 1. Total Word Count
    words = text.split()
    word_count = len(words)
    
    # 2. Sentence Count (splitting by . ! or ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    sentence_count = len(sentences)
    
    # 3. Long Word Count (words > 5 characters)
    long_words = [w for w in words if len(re.sub(r'\W+', '', w)) > 5]
    long_word_count = len(long_words)
    
    # 4. Average Word Length
    if word_count > 0:
        avg_word_len = sum(len(w) for w in words) / word_count
    else:
        avg_word_len = 0
        
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "long_word_count": long_word_count,
        "avg_word_length": round(avg_word_len, 2)
    }

# Test it with the text from your OCR
if __name__ == "__main__":
    # Load the text saved from the previous OCR step
    try:
        with open("essay_input.txt", "r") as f:
            essay_content = f.read()
            
        features = extract_surface_features(essay_content)
        print("--- Extracted Features ---")
        for key, value in features.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
            
    except FileNotFoundError:
        print("Please run the OCR script first to generate essay_input.txt")
