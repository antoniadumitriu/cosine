import os
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

# Set the directory containing your text files
corpus_dir = 'path/to/your/corpus/directory'

# Load and preprocess text files
text_data = []
filenames = []
for filename in os.listdir(corpus_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            preprocessed_text = preprocess_text(text)
            text_data.append(preprocessed_text)
            filenames.append(filename)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

# Compute cosine similarity matrix
cosine_similarity_matrix = cosine_similarity(tfidf_matrix)

# Print similarity scores
for i in range(len(filenames)):
    for j in range(len(filenames)):
        if i != j:
            print(f"Cosine similarity between {filenames[i]} and {filenames[j]}: {cosine_similarity_matrix[i][j]}")
