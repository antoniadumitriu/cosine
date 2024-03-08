import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters, convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and stem words
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)

# Load and preprocess text files
with open('file1.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()
    preprocessed_text1 = preprocess_text(text1)

with open('file2.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()
    preprocessed_text2 = preprocess_text(text2)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])

# Compute cosine similarity
cosine_similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

print(f"Cosine similarity score: {cosine_similarity_score}")
