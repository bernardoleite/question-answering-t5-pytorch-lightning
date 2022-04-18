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
for document in train_data["data"]:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            train_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
train_df = pd.DataFrame(train_all_compiled, columns = ['context', 'question', 'answer'])

print("Train Dataframe completed.")

val_all_compiled = []
for document in validation_data["data"]:
    paragraphs = document["paragraphs"]
    for para in paragraphs:
        context = para["context"]
        qas = para["qas"]
        for qa in qas:
            val_all_compiled.append([context, qa["question"], qa["answers"][0]["text"]])
validation_df = pd.DataFrame(val_all_compiled, columns = ['context', 'question', 'answer'])

print("Validation Dataframe completed.")

# In this case, test_df is the same as validation_df (test is not available from squad team)
test_df = validation_df

print("Test Dataframe completed.")
print("\n")
print("Number of train QA-Paragrah pairs: ", len(train_df))
print("Number of validation QA-Paragrah pairs: ", len(validation_df))
print("Number of test QA-Paragrah pairs: ", len(test_df))

train_df.to_pickle("../../data/squad_en_original/processed/df_train_en.pkl")
validation_df.to_pickle("../../data/squad_en_original/processed/df_validation_en.pkl")
test_df.to_pickle("../../data/squad_en_original/processed/df_test_en.pkl")
print("Pickles were generated from dataframes.")

# Code for analyzing indivual example
#print("\n")
#print(train_df['context'].iloc[75720])
#print("\n")
#print(train_df['question'].iloc[75720])
#print("\n")
#print(train_df['answer'].iloc[75720])


