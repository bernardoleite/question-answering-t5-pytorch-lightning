import pandas as pd
import nltk
import sys
import json
import uuid

MAX_LEN = 128

# shift + alt + a
import pyterrier as pt
if not pt.started():
  pt.init()

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

def top_results(document_title, question, sentences):    
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
        path_index = "C:/Users/Bernardo/Desktop/GitHubRepos/qa-transformers-pytorch/src/data/terrier_indexes/" + str(uuid.uuid4().hex) + "/"
        pd_indexer = pt.DFIndexer(path_index, overwrite=True, blocks=True)
        indexref2 = pd_indexer.index(df["sentences"], df["docno"])

        index = pt.IndexFactory.of(indexref2)
        #print(index.getCollectionStatistics().toString())

        question_processed = question.replace("?", "")
        question_processed = question_processed.replace("'", "")
        question_processed = question_processed.replace("/", "")
        topics = pd.DataFrame([["1", question_processed]], columns=['qid', 'query'])

        retr = pt.BatchRetrieve(index, wmodel="BM25", num_results=1000)
        res = retr.transform(topics)
    except Exception as e:
        print("Exception!!! ", e)
        res = -1

    return res

def get_text_info(sents_indexes, sentences, question):
    sentences_final = []
    for index, sent in enumerate(sentences):
        if index in sents_indexes:
            sentences_final.append(sent)

    if len(sentences_final) == 0:
        print("Nr of final sentences cant be 0!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit()

    tokenized_ovf_inputs = tokenize_example(question, ' '.join(sentences_final))

    return len(tokenized_ovf_inputs), sentences_final

def handle_qa(document_title, context, qa_id, question):
    sentences_final = []
    sentences = nltk.sent_tokenize(context) # this gives us a list of sentences
    res = top_results(document_title, question, sentences)
    if type(res) == int and res == -1:
        print("Error in value returned!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sentences_final = sentences
        #sys.exit()
    else:
        if len(res.docno) > 0:
            list_top_results = [int(elem) for elem in res.docno] # [0,15,4,3,...] real indexes of sentences ordered by tf-idf, bm25, etc, ...
            sents_indexes = list_top_results
            
            num_ovf_tokens = 10000
            while num_ovf_tokens != 0 and len(sents_indexes) > 0:
                num_ovf_tokens, sentences_final = get_text_info(sents_indexes, sentences, question)
                if num_ovf_tokens > 0:
                    del sents_indexes[-1]
            if len(list_top_results) != len(sents_indexes):
                print("!!!!!!!!!!!!!!!!!Aconteceu isto!!!!!!!!!!!!!!!!!!!!!!")
                print(list_top_results)
                print(sents_indexes)
                print("\n")
                sys.exit()

        else:
            sentences_final = sentences
            print("len(res.docno) == 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if qa_id == '56beace93aeaaa14008c91e2':
        print(sentences,"\n")
        print(sentences_final)
        sys.exit()

    print("Nr of original sentences: ", len(sentences))
    print("Nr of final sentences: ", len(sentences_final))

    return ' '.join(sentences_final)

 # Loading JSON files
file_to_open = '../../data/squad_en_original/raw/dev-v1.1-filtered-' + str(MAX_LEN) + '.json'
with open(file_to_open) as val_json_file:
    val_data = json.load(val_json_file)

val_all_compiled = []
for document in val_data["data"]:               # data -> document
    document_title = document["title"]          # document -> title
    paragraphs = document["paragraphs"]         # document -> paragraphs
    for para in paragraphs:
        context = para["context"]               # document -> paragraphs[i] -> context
        qas = para["qas"]                       # document -> paragraphs[i] -> qas
        for qa in qas:
            qa_id = qa["id"]                    # document -> paragraphs[i] -> qas[i] -> id
            question = qa["question"]           # document -> paragraphs[i] -> qas[i] -> question
            answer = qa["answers"][0]["text"]   # document -> paragraphs[i] -> qas[i] -> answer

            new_context = handle_qa(document_title, context, qa_id, question)
    
            val_all_compiled.append([document_title, new_context, qa_id, question, answer])

df_validation_en_filtered = pd.DataFrame(val_all_compiled, columns = ['document_title', 'context', 'qa_id', 'question', 'answer'])

#df_validation_en_filtered.to_pickle("../../data/squad_en_original/processed/df_validation_en_filtered_128_bm25.pkl")
print("Pickles were generated from dataframes.")