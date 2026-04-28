import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def train_model(csv_file):
    # 1. Load the dataset (CSV with features and a 'score' column)
    df = pd.read_csv(csv_file)
    
    # 2. Separate features (X) and target scores (y)
    # Make sure your CSV columns match the names in processor.py
    X = df.drop('score', axis=1)
    y = df['score']
    
    # 3. Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Scaling (SVR is very sensitive to scale!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Initialize and Train SVR
    # C and epsilon are hyperparameters mentioned in the research paper
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    # 6. Save the model and the scaler for later use
    joblib.dump(model, 'essay_scoring_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Training complete! Model saved as 'essay_scoring_model.pkl'")
    print(f"Model Accuracy (R2 Score): {model.score(X_test_scaled, y_test):.2f}")

if __name__ == "__main__":
    # If you don't have a CSV, you'll need to create one first
    # using the features from multiple main.py runs.
    train_model('training_data.csv')
