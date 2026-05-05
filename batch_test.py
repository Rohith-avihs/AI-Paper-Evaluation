import pandas as pd
from sentence_transformers import SentenceTransformer
from processor import get_feature_vector
import numpy as np

# Load lightweight model for speed
model = SentenceTransformer('all-MiniLM-L6-v2')

def run_headless_test(csv_path, reference_text, max_score=12):
    df = pd.read_csv(csv_path)
    # Filter for set 1 if necessary
    if 'essay_set' in df.columns:
        df = df[df['essay_set'] == 1]

    ref_emb = model.encode([reference_text])[0]
    scores = []

    for idx, row in df.iterrows():
        # Score Logic
        stu_emb = model.encode([row['essay']])[0]
        sim = np.dot(ref_emb, stu_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(stu_emb))
        
        # Combined AI Score
        ai_score = round(sim * max_score, 2)
        scores.append(ai_score)
    
    df['AI_Predicted_Score'] = scores
    df['Absolute_Error'] = abs(df['domain1_score'] - df['AI_Predicted_Score'])
    
    print(f"Average Error: {df['Absolute_Error'].mean()}")
    df.to_csv("test_results.csv", index=False)

if __name__ == "__main__":
    # Ensure you have a small 'asap_sample.csv' in the folder
    # run_headless_test("asap_sample.csv", "Your gold standard answer here")
    pass
