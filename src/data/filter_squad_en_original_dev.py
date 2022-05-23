import pandas as pd
import json
import sys
sys.path.append('../')

MAX_LEN = 256

from transformers import (
    T5Tokenizer
)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def tokenize_example(question, context):
    # tokenize inputs
    tokenized_inputs = t5_tokenizer(
        question,
        context,
        truncation = 'only_second',
        add_special_tokens=True,
        return_overflowing_tokens = True, # if (source_encoding.num_truncated_tokens.item() > 0): !!!!!!!! (future)
        max_length=MAX_LEN, 
        padding='max_length', 
        return_tensors="pt"
    )
    return tokenized_inputs.overflowing_tokens[0].tolist()

 # Loading JSON files
with open('../../data/squad_en_original/raw/dev-v1.1.json') as val_json_file:
    val_data = json.load(val_json_file)

count_qas_filtered = 0
qas_filtered_ids = []

#val_all_compiled = []
for document in val_data["data"]:  # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    for para in paragraphs:
        context = para["context"]               # document -> paragraphs[i] -> context
        qas = para["qas"]                       # document -> paragraphs[i] -> qas
        for qa in qas:
            qa_id = qa["id"]                    # document -> paragraphs[i] -> qas[i] -> id
            question = qa["question"]           # document -> paragraphs[i] -> qas[i] -> question
            answer = qa["answers"][0]["text"]   # document -> paragraphs[i] -> qas[i] -> answer

            tokenized_inputs = tokenize_example(question, context)

            if len(tokenized_inputs) > 0:
                count_qas_filtered = count_qas_filtered + 1
                qas_filtered_ids.append(qa_id)
    
            #val_all_compiled.append([document_title, context, qa_id, question, answer])

#val_df = pd.DataFrame(val_all_compiled, columns = ['document_title', 'context', 'qa_id', 'question', 'answer'])

val_data_filtered = val_data

count_new_qas = 0

for document in val_data_filtered["data"]:
    for para in document["paragraphs"] :
        qas = para["qas"]
        new_qas = [qa for qa in qas if qa["id"] in qas_filtered_ids]
        count_new_qas = count_new_qas + len(new_qas)
        para["qas"] = new_qas

print(count_new_qas)

with open('../../data/squad_en_original/raw/dev-v1.1-filtered-'+str(MAX_LEN)+'.json', 'w') as f:
    json.dump(val_data_filtered, f)