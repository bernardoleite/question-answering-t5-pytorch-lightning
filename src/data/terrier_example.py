import pandas as pd
import nltk
import json
import sys
sys.path.append('../')

from transformers import (
    T5Tokenizer
)
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# shift + alt + a
""" import pyterrier as pt
if not pt.started():
  pt.init() """

def close_sentences(document_title, question, sentences):    
    if len(sentences) > 0:
        nr_sents = len(sentences)
        list_docno_int = list(range(0, nr_sents))
        list_docno_str = list(map(str, list_docno_int)) # mapping list of int to list of str
    else:
        print("Error. Number of sentences is 0!")
        return -1

    df = pd.DataFrame({
        'docno': list_docno_str,
        'sentences': sentences
    })

    try:
        pd_indexer = pt.DFIndexer("./pd_index", overwrite=True, blocks=True)
        indexref2 = pd_indexer.index(df["sentences"], df["docno"])

        index = pt.IndexFactory.of(indexref2)
        #print(index.getCollectionStatistics().toString())

        #new_answer = answer.replace("'", "")
        topics = pd.DataFrame([["1", question]], columns=['qid', 'query'])

        retr = pt.BatchRetrieve(index, wmodel="TF_IDF", num_results=100)
        res = retr.transform(topics)
    except Exception as e:
        print("Exception!!! ", e)
        res = -1

    return res

def tokenize_example(question, context):
    # tokenize inputs
    tokenized_inputs = t5_tokenizer(
        question,
        context,
        truncation = 'only_second',
        add_special_tokens=True,
        return_overflowing_tokens = True, # if (source_encoding.num_truncated_tokens.item() > 0): !!!!!!!! (future)
        max_length=512, 
        padding='max_length', 
        return_tensors="pt"
    )
    
    return tokenized_inputs.overflowing_tokens[0].tolist()

 # Loading JSON files
with open('../../data/squad_en_original/raw/dev-v1.1.json') as val_json_file:
    val_data = json.load(val_json_file)

count_qas_filtered = 0
qas_filtered = []

val_all_compiled = []
for document in val_data["data"]:  # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    for para in paragraphs:
        context = para["context"]               # document -> paragraphs[i] -> context
        #context  = '20th Century Fox, Lionsgate, Paramount Pictures, Universal Studios and Walt Disney Studios paid for movie trailers to be aired during the Super Bowl. Fox paid for Deadpool, X-Men: Apocalypse, Independence Day: Resurgence and Eddie the Eagle, Lionsgate paid for Gods of Egypt, Paramount paid for Teenage Mutant Ninja Turtles: Out of the Shadows and 10 Cloverfield Lane, Universal paid for The Secret Life of Pets and the debut trailer for Jason Bourne and Disney paid for Captain America: Civil War, The Jungle Book and Alice Through the Looking Glass.[citation needed]'
        sentences = nltk.sent_tokenize(context) # this gives us a list of sentences
        qas = para["qas"]                       # document -> paragraphs[i] -> qas
        for qa in qas:
            qa_id = qa["id"]                    # document -> paragraphs[i] -> qas[i] -> id
            question = qa["question"]           # document -> paragraphs[i] -> qas[i] -> question
            answer = qa["answers"][0]["text"]   # document -> paragraphs[i] -> qas[i] -> answer
            tokenized_inputs = tokenize_example(question,context)
            if len(tokenized_inputs) > 0:
                count_qas_filtered = count_qas_filtered + 1
                qas_filtered.append(qa_id)
            #res = close_sentences(document_title, question, sentences)
            #if type(res) == int and res == -1:
                #pass
            #else:
                #if len(res.docno) >= 3:
                    #print(res)
            val_all_compiled.append([document_title, context, qa_id, question, answer])
val_df = pd.DataFrame(val_all_compiled, columns = ['document_title', 'context', 'qa_id', 'question', 'answer'])

val_data_filtered = val_data

for document in val_data_filtered["data"]:
    for para in document["paragraphs"] :
        qas = para["qas"]
        new_qas = [qa for qa in qas if qa["id"] in qas_filtered]
        para["qas"] = new_qas

with open('../../data/squad_en_original/raw/dev-v1.1-filtered.json', 'w') as f:
    json.dump(val_data_filtered, f)