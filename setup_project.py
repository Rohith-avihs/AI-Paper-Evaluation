import nltk

def setup():
    print("Downloading NLTK models...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    print("Setup complete.")

if __name__ == "__main__":
    setup()
