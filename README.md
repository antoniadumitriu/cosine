# Cosine Similarity

This repository contains Python scripts for calculating the cosine similarity between text documents. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is often used in text analysis, information retrieval, and data mining tasks to determine the similarity between documents, queries, and other text data.

## Scripts

1. **cosine.py**
   - Calculates the cosine similarity between two hard-coded text strings.
   - Demonstrates the basic implementation of cosine similarity calculation.

2. **cosine_2text.py**
   - Calculates the cosine similarity between two text files.
   - Allows you to specify the file paths for the two files to be compared.

3. **cosine_corpus.py**
   - Calculates the cosine similarity between all text files in a specified directory.
   - Generates a similarity matrix showing the similarity scores between each pair of files.

## Prerequisites

- Python 3.x
- NLTK (Natural Language Toolkit) library

To install NLTK, run:

```
pip install nltk
```

After installing NLTK, you'll need to download the required data packages by running the following in your Python interpreter:

```python
import nltk
nltk.download('punkt')
```
