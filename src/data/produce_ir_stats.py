#https://stackoverflow.com/questions/56707363/how-to-get-top-n-terms-with-highest-tf-idf-score-big-sparse-matrix

import json
import sys
sys.path.append('../')

 # Loading JSON files
with open('../../data/squad_en_original/raw/train-v1.1.json', encoding='utf-8') as train_json_file:
    train_data = json.load(train_json_file)

corpus = []

train_all_compiled = []
for document in train_data["data"]:             # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    document_paragraphs = []

    for para in paragraphs:
        document_paragraphs.append(para["context"])
    document_text = ' '.join(document_paragraphs)
    corpus.append(document_text)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
feature_array = vectorizer.get_feature_names()

top_n = 10

print('tf_idf scores: \n', sorted(list(zip(vectorizer.get_feature_names(), X.sum(0).getA1())), key=lambda x: x[1], reverse=True)[:top_n])

# tf_idf scores : 
# [('document', 1.4736296010332683), ('check', 0.6227660078332259), ('like', 0.6227660078332259)]
print("\n")

print('idf values: \n', sorted(list(zip(feature_array,vectorizer.idf_,)), key = lambda x: x[1], reverse=True)[:top_n])

# idf values: 
#  [('aim', 1.6931471805599454), ('capture', 1.6931471805599454), ('check', 1.6931471805599454)]
print("\n")

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
feature_array = vectorizer.get_feature_names()
print('Frequency: \n', sorted(list(zip(vectorizer.get_feature_names(), X.sum(0).getA1())), key=lambda x: x[1], reverse=True)[:top_n])

# Frequency: 
#  [('document', 2), ('aim', 1), ('capture', 1)]