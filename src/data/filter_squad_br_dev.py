import pandas as pd
import json
import sys
sys.path.append('../')

 # Loading JSON files
with open('../../data/squad_br_v2/raw/dev-v1.1-pt.json') as val_json_file:
    val_data = json.load(val_json_file)

qas_filtered_ids = []
count_qas_filtered = 0

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

            if answer in context:
                count_qas_filtered = count_qas_filtered + 1
                qas_filtered_ids.append(qa_id)

val_data_filtered = val_data

count_new_qas = 0

for document in val_data_filtered["data"]:
    for para in document["paragraphs"] :
        qas = para["qas"]
        new_qas = [qa for qa in qas if qa["id"] in qas_filtered_ids]
        count_new_qas = count_new_qas + len(new_qas)
        para["qas"] = new_qas

with open('../../data/squad_br_v2/raw/dev-v1.1-filtered-YES-ANSWER.json', 'w') as f:
    json.dump(val_data_filtered, f)

print(count_qas_filtered, count_new_qas)