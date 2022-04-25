from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    T5Tokenizer
    )

import argparse
import sys
sys.path.append('../')

from models import T5FineTuner

import torch
import pandas as pd
import utils

QUESTION = 'If Roman numerals were used, what would Super Bowl 50 have been called?'
CONTEXT = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""
GROUND_TRUTH_ANSWERS = []

def generate(args, device, qamodel: T5FineTuner, tokenizer: T5Tokenizer,  question: str, context: str) -> str:

    source_encoding = tokenizer(
        question,
        context,
        max_length=args.max_len_input,
        padding='max_length',
        truncation = 'only_second',
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Put this in GPU (faster than using cpu)
    input_ids = source_encoding['input_ids'].to(device)
    attention_mask = source_encoding['attention_mask'].to(device)

    generated_ids = qamodel.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=args.num_return_sequences, # defaults to 1
        num_beams=args.num_beams, # defaults to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! myabe experiment with 5
        max_length=args.max_len_output,
        repetition_penalty=args.repetition_penalty, # defaults to 1.0, #last value was 2.5
        length_penalty=args.length_penalty, # defaults to 1.0
        early_stopping=True, # defaults to False
        use_cache=True
    )

    preds_text = []
    for pred_tokens_ids in generated_ids:
        pred_text = tokenizer.decode(pred_tokens_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds_text.append(pred_text)

    return preds_text

def run(args):
    # Load args (needed for model init) and log json
    params_dict = dict(
        checkpoint_model_path = args.checkpoint_model_path,
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        num_beams = args.num_beams,
        num_return_sequences = args.num_return_sequences,
        repetition_penalty = args.repetition_penalty,
        length_penalty = args.length_penalty,
    )
    params = argparse.Namespace(**params_dict)

    # Load T5 base Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    # Load T5 base Model
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Load T5 fine-tuned model for QA
    checkpoint_model_path = args.checkpoint_model_path
    qamodel = T5FineTuner.load_from_checkpoint(checkpoint_model_path, hparams=params, t5model=t5_model, t5tokenizer=t5_tokenizer)

    # Put model in freeze() and eval() model. Not sure the purpose of freeze
    # Not sure if this should be after or before changing device for inference.
    qamodel.freeze()
    qamodel.eval()

    # Put model in gpu (if possible) or cpu (if not possible) for inference purpose
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qamodel = qamodel.to(device)
    print ("Device for inference:", device)

    predictions = generate(args, device, qamodel, t5_tokenizer, QUESTION, CONTEXT)

    output_predictions(predictions)

def output_predictions(predictions):
    table_results = pd.DataFrame(columns=['model_answer_prediction', 'model_rank_score', 'f1_score', 'exact_match_score'])
    print("Text: ", CONTEXT, "\n")
    print("Question: ", QUESTION, "\n")
    for index, pred in enumerate(predictions):
        if len(GROUND_TRUTH_ANSWERS) > 0:
            f1_score = utils.metric_max_over_ground_truths(utils.get_f1_score, pred, GROUND_TRUTH_ANSWERS)
            exact_match_score = utils.metric_max_over_ground_truths(utils.get_exact_match_score, pred, GROUND_TRUTH_ANSWERS)
        else: # there are no ground_truth answers available
            f1_score = -1
            exact_match_score = -1
        table_results = table_results.append({'model_answer_prediction': pred, 'model_rank_score': index+1, 'f1_score': round(f1_score, 3), 'exact_match_score': exact_match_score*1}, ignore_index=True)
        #print('BEAM: %d | ANSWER: %s | F1-SCORE: %.3f | EXACT-MATCH: %d'  % (index+1, pred, f1_score, exact_match_score))
    print(table_results.to_string(index=False))
    #save in csv...
    #table_results.to_csv("out.csv", sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Inference for generating text using models.')

    # Add arguments
    parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default="../../checkpoints/qa_en_t5_base_512_96_32_10_seed_42/model-epoch=00-val_loss=0.32.ckpt", required=False, help='Model folder checkpoint path.')

    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-base", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="t5-base", required=False, help='Tokenizer name.')

    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=96, required=False, help='Max len output for encoding.')

    parser.add_argument('-nb','--num_beams', type=int, metavar='', default=15, required=False, help='Number of beams.')
    parser.add_argument('-nrs','--num_return_sequences', type=int, metavar='', default=15, required=True, help='Number of returned sequences.')
    parser.add_argument('-rp','--repetition_penalty', type=float, metavar='', default=1.0, required=False, help='Repetition Penalty.')
    parser.add_argument('-lp','--length_penalty', type=float, metavar='', default=1.0, required=False, help='Length Penalty.')

    # Parse arguments
    args = parser.parse_args()

    # Start tokenization, encoding and generation
    run(args)