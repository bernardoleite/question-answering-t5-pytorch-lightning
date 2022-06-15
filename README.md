# Question Answering with T5 model using 🤗Transformers and Pytorch Lightning

## Overview
Question Answering (QA) is the task of automatically answering questions given a paragraph, document, or collection. This open-source project aims to provide simplified training & inference routines, and QA fine-tuned models to facilitate and speed up research and experimentation within the QA task. Initial experiments use the T5 model and the Stanford Question Answering Dataset (SQuAD) v1.1. Here the goal is to **generate an answer** given the pair **(question, text)**.

## Example
_...As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50..._

**Question**: If Roman numerals were used, what would Super Bowl 50 have been called?

**Generated Answer**: Super Bowl L

## Main Features
* Training, inference and evaluation scripts for QA
* Fine-tuned QA T5 models for both English and Portuguese

## Prerequisites
```bash
Python 3
```

## Installation and Configuration
1. Clone this project:
    ```python
    git clone https://github.com/bernardoleite/question-answering-t5-pytorch-lightning
    ```
2. Install the Python packages from [requirements.txt](https://github.com/bernardoleite/question-generation-portuguese/blob/main/requirements.txt). If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:
    ```bash
    cd question-answering-t5-pytorch-lightning/
    pip install -r requirements.txt
    ```

## Usage
You can use this code for **data preparation**, **training**, **inference/predicting** (full corpus or individual sample), and **evaluation**.

### Data preparation
Current experiments use the SQuAD v1.1 dataset for English (original) and Portuguese (machine-translated) versions. So the next steps are specifically intended to preparing this dataset, but the same approach is applicable to other data types.
* Example for preparing the **English** (original) SQuAD v1.1 dataset:
1.  Download `train-v1.1.json` and `dev-v1.1.json` data files from [here](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset).
2.  Create `squad_en_original/raw/` and `squad_en_original/processed/` foders inside `data/` and copy previous files to `data/squad_en_original/raw/`.
3.  Go to `src/data`. By running `src/data/pre_process_squad_en_original.py` the following dataframes (pickle format) will be created inside `data/squad_en_original/processed/`: `df_train_en.pkl` and `df_validation_en.pkl`.

* Example for preparing the **Portuguese** (machine-translated) SQuAD v1.1 dataset:
1.  Download `train-v1.1-pt.json` and `dev-v1.1-pt.json` data from [here](https://github.com/nunorc/squad-v1.1-pt).
2.  Create `squad_br_v2/raw` and `squad_br_v2/processed` folders inside `data/` and copy previous files to `data/squad_br_v2/raw/`.
3.  Go to `src/data`. By running `src/data/pre_process_squad_br.py` and then `src/data/pre_process_squad_br_processed.py` the following dataframes (pickle format) will be created inside `data/squad_br_v2/processed/`: `df_train_br.pkl` and `df_validation_br.pkl`.

**Important note**: Regardless of the data type, make sure the dataframe columns follow this scheme: [**document_title**, **context**, **qa_id**, **question**, **answer**].

### Training 
1.  Go to `src/model_qa`. The file `train.py` is responsible for the training routine. Type the following command to read the description of the parameters:
    ```bash
    python train.py -h
    ```
    You can also run the example training script (linux and mac) `train_qa_en_t5_base_512_96_32_10.sh`:
    ```bash
    bash train_qa_en_t5_base_512_96_32_10.sh
    ```
    The previous script will start the training routine with predefined parameters:
    ```python
    #!/usr/bin/env bash

    for ((i=42; i <= 42; i++))
    do
        CUDA_VISIBLE_DEVICES=1 python train.py \
        --dir_model_name "qa_en_t5_base_512_96_32_10_seed_${i}" \
        --model_name "t5-base" \
        --tokenizer_name "t5-base" \
        --train_df_path "../../data/squad_en_original/processed/df_train_en.pkl" \
        --validation_df_path "../../data/squad_en_original/processed/df_validation_en.pkl" \
        --test_df_path "../../data/squad_en_original/processed/df_test_en.pkl" \
        --max_len_input 512 \
        --max_len_output 96 \
        --batch_size 32 \
        --max_epochs 10 \
        --patience 3 \
        --optimizer "AdamW" \
        --learning_rate 0.0001 \
        --epsilon 0.000001 \
        --num_gpus 1 \
        --seed_value ${i}
    done
    ```

2. In the end, all model information is available at `checkpoints/checkpoint-name`. The information includes models checkpoints for each epoch (`*.ckpt` files), tensorboard logs (`tb_logs/`) and csv logs (`csv_logs/`).

3. Previous steps also apply to `train_qa_br_v2_ptt5_base_512_96_32_10.sh` for the training routine in Portuguese.

### Inference (full corpus)
1.  Go to `src/model_qa`. The file `inference_corpus.py` is responsible for the inference routine (full corpus) given a certain **model checkpoint**. Type the following command to read the description of the parameters:
    ```bash
    python inference_corpus.py -h
    ```
    You can also run the example inference corpus script (linux and mac) `inference_corpus_qa_en_t5_base_512_96_10.sh`:
    ```bash
    bash inference_corpus_qa_en_t5_base_512_96_10.sh
    ```
    The previous script will start the inference routine with predefined parameters for the model checkpoint `model-epoch=00-val_loss=0.32.ckpt`:
    ```python
    #!/usr/bin/env bash

    for ((i=42; i <= 42; i++))
    do
        CUDA_VISIBLE_DEVICES=1 python inference_corpus.py \
        --checkpoint_model_path "../../checkpoints/qa_en_t5_base_512_96_32_10_seed_42/model-epoch=00-val_loss=0.32.ckpt" \
        --predictions_save_path "../../predictions/qa_en_t5_base_512_96_32_10_seed_42/model-epoch=00-val_loss=0.32/" \
        --test_df_path "../../data/squad_en_original/processed/df_validation_en.pkl" \
        --model_name "t5-base" \
        --tokenizer_name "t5-base" \
        --batch_size 32 \
        --max_len_input 512 \
        --max_len_output 96 \
        --num_beams 5 \
        --num_return_sequences 1 \
        --repetition_penalty 1 \
        --length_penalty 1 \
        --seed_value ${i}
    done
    ```
2. In the end, predictions will be available at `predictions/checkpoint-name`. The folder contains model predictions (`predictions.json`), and parameters (`params.json`).

3. Previous steps also apply to `inference_corpus_qa_br_v2_ptt5_base_512_96_10.sh` for the inference routine in Portuguese.

### Inference (individual sample)
Go to `src/model_qa`. The file `inference_example.py` is responsible for the inference routine (individual sample) given a certain **model checkpoint**, **CONTEXT** and **QUESTION**. Type the following command to read the description of the parameters:

```bash
python inference_example.py -h
```
Example/Demo:

1.  Change **QUESTION** and **CONTEXT** variables in `inference_example.py`:
    ```python
    QUESTION = 'If Roman numerals were used, what would Super Bowl 50 have been called?'
    CONTEXT = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""
    ```

2.  Run `inference_example.py` (e.g., using `model-epoch=00-val_loss=0.32.ckpt` as model checkpoint and 2 for `num_return_sequences`).

3.  See output:
    ```python
    QUESTION:  If Roman numerals were used, what would Super Bowl 50 have been called?

    Answer prediction for model_returned_sequence 0: Super Bowl L
    Answer prediction for model_returned_sequence 1: Super Bowl L"
    ```

### Evaluation 
To do.

## Issues and Usage Q&A
To ask questions, report issues or request features, please use the GitHub Issue Tracker.

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks in advance!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
### Project
This project is released under the **MIT** license. For details, please see the file [LICENSE](https://github.com/bernardoleite/question-answering-t5-pytorch-lightning/blob/main/requirements.txt) in the root directory.

### Commercial Purposes
A commercial license may also be available for use in industrial projects, collaborations or distributors of proprietary software that do not wish to use an open-source license. Please contact the author if you are interested.

## Acknowledgements
This project is inspired by/based on the implementations of [Venelin Valkov](https://www.youtube.com/watch?v=r6XY80Z9eSA&t=1994s), [Ramsri Golla](https://www.udemy.com/course/question-generation-using-natural-language-processing/), [Suraj Patil](https://github.com/patil-suraj/question_generation) and [Kristiyan Vachev](https://github.com/KristiyanVachev/Question-Generation).

## Contacts
* Bernardo Leite, bernardo.leite@fe.up.pt