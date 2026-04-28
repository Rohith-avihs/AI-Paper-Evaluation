import joblib
import sys
import os
from processor import extract_text, get_feature_vector

def load_ai_model():
    """Loads the trained SVR model and the feature scaler."""
    try:
        model = joblib.load('essay_scoring_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Error: Model or Scaler not found. Run trainer.py first!")
        sys.exit(1)

def predict_score(image_path):
    # 1. Load the Brain
    model, scaler = load_ai_model()

    # 2. Extract Text and Features (from processor.py)
    print(f"Reading image: {image_path}...")
    text = extract_text(image_path)
    if not text.strip():
        return "Could not extract text from image."
    
    features_dict = get_feature_vector(text)
    
    # 3. Convert dict to a flat list for the AI (Feature Vector)
    # Important: Order must match the order used in training!
    feature_values = [
        features_dict['word_count'],
        features_dict['sent_count'],
        features_dict['long_word_count'],
        features_dict['avg_word_len'],
        features_dict['noun_count'],
        features_dict['verb_count'],
        features_dict['adj_count']
    ]

    # 4. Scale and Predict
    features_scaled = scaler.transform([feature_values])
    predicted_score = model.predict(features_scaled)

    return round(predicted_score[0], 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py your_test_image.jpg")
    else:
        img = sys.argv[1]
        score = predict_score(img)
        print(f"\n--- Evaluation Result ---")
        print(f"Predicted Score: {score} / 20")
