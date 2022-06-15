import pandas as pd

import json
import sys
sys.path.append('../')

 # Loading JSON files
with open('../../data/squad_en_original/raw/train-v1.1.json') as train_json_file:
    train_data = json.load(train_json_file)

with open('../../data/squad_en_original/raw/dev-v1.1.json') as dev_json_file:
    validation_data = json.load(dev_json_file)

train_all_compiled = []
for document in train_data["data"]:             # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    for para in paragraphs:
        context = para["context"]               # document -> paragraphs[i] -> context
        qas = para["qas"]                       # document -> paragraphs[i] -> qas
        for qa in qas:
            qa_id = qa["id"]                    # document -> paragraphs[i] -> qas[i] -> id
            question = qa["question"]           # document -> paragraphs[i] -> qas[i] -> question
            answer = qa["answers"][0]["text"]   # document -> paragraphs[i] -> qas[i] -> answer
            train_all_compiled.append([document_title, context, qa_id, question, answer])
train_df = pd.DataFrame(train_all_compiled, columns = ['document_title', 'context', 'qa_id', 'question', 'answer'])

print("Train Dataframe completed.")

validation_all_compiled = []
for document in validation_data["data"]:             # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    for para in paragraphs:
        context = para["context"]               # document -> paragraphs[i] -> context
        qas = para["qas"]                       # document -> paragraphs[i] -> qas
        for qa in qas:
            qa_id = qa["id"]                    # document -> paragraphs[i] -> qas[i] -> id
            question = qa["question"]           # document -> paragraphs[i] -> qas[i] -> question
            answer = qa["answers"][0]["text"]   # document -> paragraphs[i] -> qas[i] -> answer
            validation_all_compiled.append([document_title, context, qa_id, question, answer])
validation_df = pd.DataFrame(validation_all_compiled, columns = ['document_title', 'context', 'qa_id', 'question', 'answer'])

print("Validation Dataframe completed.")

print("\n")
print("Number of train QA-Paragrah pairs: ", len(train_df))
print("Number of validation QA-Paragrah pairs: ", len(validation_df))

train_df.to_pickle("../../data/squad_en_original/processed/df_train_en.pkl")
validation_df.to_pickle("../../data/squad_en_original/processed/df_validation_en.pkl")
print("Pickles were generated from dataframes.")

# Code for analyzing indivual example
#print("\n")
#print(train_df['context'].iloc[75720])
#print("\n")
#print(train_df['question'].iloc[75720])
#print("\n")
#print(train_df['answer'].iloc[75720])


