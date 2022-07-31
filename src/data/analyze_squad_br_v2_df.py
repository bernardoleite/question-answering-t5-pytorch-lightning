import pandas as pd
import sys
sys.path.append('../')

squad_br_v2_train_df_path = '../../data/squad_br_v2/dataframe/df_train_br.pkl'
squad_br_v2_val_df_path = '../../data/squad_br_v2/dataframe/df_validation_br.pkl'

train_df = pd.read_pickle(squad_br_v2_train_df_path)
val_df = pd.read_pickle(squad_br_v2_val_df_path)

answer_in_train = 0
for index, row in train_df.iterrows():
    if row['answer'] in row['context']:
        answer_in_train = answer_in_train + 1
print("Num of gt answers in train: %d from %d total answers" % (answer_in_train, len(train_df)))

answer_in_val = 0
for index, row in val_df.iterrows():
    if row['answer'] in row['context']:
        answer_in_val = answer_in_val + 1
print("Num of gt answers in val: %d from %d total answers" % (answer_in_val, len(val_df)))