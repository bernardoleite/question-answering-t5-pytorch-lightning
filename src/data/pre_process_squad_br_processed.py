import pandas as pd
from datasets import load_dataset
import json
import sys
sys.path.append('../')

squad_br_train_path = '../../data/squad_br_v2/processed/processed-train-v1.1-pt.json'
squad_br_val_path = '../../data/squad_br_v2/processed/processed-dev-v1.1-pt.json'

# following tutorial...
datasets = load_dataset('json', 
                        data_files={'train': squad_br_train_path, 'validation': squad_br_val_path}, 
                        field='data')

train_all_compiled = []
val_all_compiled = []

for elem in datasets["train"]:
    document_title = elem["title"]
    context = elem["context"]
    qa_id = elem["id"]
    question = elem["question"]
    answer = elem["answers"]["text"][0]

    train_all_compiled.append([document_title, context, qa_id, question, answer])

# Save to dataframes
train_df = pd.DataFrame(train_all_compiled, columns = ['document_title','context', 'qa_id', 'question', 'answer'])
print("Train Dataframe completed.")

# Get validation data
for elem in datasets["validation"]:
    document_title = elem["title"]
    context = elem["context"]
    qa_id = elem["id"]
    question = elem["question"]
    answer = elem["answers"]["text"][0]

    val_all_compiled.append([document_title, context, qa_id, question, answer])

validation_df = pd.DataFrame(val_all_compiled, columns = ['document_title','context', 'qa_id', 'question', 'answer'])

print("Validation Dataframe completed.")
print("\n")
print("Number of train QA-Paragrah pairs: ", len(train_df))
print("Number of validation QA-Paragrah pairs: ", len(validation_df))

train_df.to_pickle("../../data/squad_br_v2/processed/df_train_br.pkl")
validation_df.to_pickle("../../data/squad_br_v2/processed/df_validation_br.pkl")

print("\n","Pickles were generated from dataframes.")