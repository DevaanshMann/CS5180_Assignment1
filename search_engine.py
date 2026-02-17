#-------------------------------------------------------------
# AUTHOR: Devaansh Mann
# FILENAME: search_engine.py
# SPECIFICATION:Binary term-weight search engine using unigrams + bigrams,
#               Porter stemming, stop word removal, and dot-product scoring.
# FOR: CS 5180- Assignment #1
# TIME SPENT: ~ 1.5 - 2 hours
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
# --> add your Python code here

print("Document:", documents)

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------
# --> add your Python code here

stop_words = ["i", "a", "and", "they", "their", "the"]

stemmer = PorterStemmer()

def stemming_tokenizer(text):
    tokens = text.lower().split()
    cleaned = []
    for token in tokens:
        token = token.strip(".,!?;:'\"")
        if token and token not in stop_words:
            cleaned.append(stemmer.stem(token))
    return cleaned

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = stemming_tokenizer,
    stop_words = None,
    ngram_range = (1, 2),
    binary = True
)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------
# --> add your Python code here

vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
# --> add your Python code here

query = ["I love dogs"]
query_vector = vectorizer.transform(query)

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here

doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray().tolist()[0]

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------

scores = []
# --> add your Python code here
for i, doc_vec in enumerate(doc_vectors):
    score = sum(q * d for q, d in zip(query_vector, doc_vec))
    scores.append((i + 1, score))

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------

ranking = []
# --> add your Python code here
ranking = sorted(scores, key=lambda x: x[1], reverse=True)
for rank, (doc_num, score) in enumerate(ranking, start=1):
    print(f"Rank {rank}: d{doc_num} → score = {score} | {documents[doc_num - 1]}")