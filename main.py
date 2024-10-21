# Promising false friends pairs:
# Note: From only starting the search for false friends language pairs, words that also sound the same but have a different meaning are also catagorized as false friends.
# English and Spanish have lots of false friends words that fall under that catagory, but, the words themselves seem to be spelled quite differently
# English-German https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_German
# English-French https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_French
from fastjsonschema.indent import indent
from transformers import pipeline
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets as ds
import pandas as pd
from IPython.display import display
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_dataset(ds_name, split, config=None):
    """
    Loads the HuggingFace dataset
    :param ds_name: the name of the dataset in huggingface format
    :param split: the chosen split (train, validation or test)
    :param config: the chosen configuration
    :return: the dataset
    """
    if config is None:
        return ds.load_dataset(ds_name, split=split)
    else:
        return ds.load_dataset(ds_name, config, split=split)


def ds_to_json(data, file_name):
    """
    Saves the Dataset to csv format
    :param data: the dataset
    :param file_name: the file name
    :return:
    """
    data.to_json(file_name, orient='records', lines=False, indent=4)


def lower_case_data(dataset):
    """
    Lower cases all the data in the dataset
    :param dataset: the huggingface dataset
    :return: the lowercased dataset
    """
    for key in dataset:
        dataset[key] = dataset[key].lower()
    return dataset

def ff_filter(row, col1, col2):
    return row[col1] == row[col2]
    
def generate_sentence(word, language, generator, sentence_length=50):
    prompt = f"Write a sentence in {language} using the word '{word}'"
    # response is a list of dictionaries
    response = generator(prompt=prompt, max_length=sentence_length)[0]['generated_text']
    return response


if __name__ == '__main__':
    
    unk_token = "<UNK>"  # token for unknown words
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens
    preprocess_and_dl = False
    # Loading and Saving the different datasets: english dataset, german dataset, false friends english-german
    # Preprocessing on the datasets
    if preprocess_and_dl:
        row_limit = 50000
        split = f"train[:{row_limit}]"
        
        en_ds = get_dataset("Salesforce/wikitext", split, "wikitext-103-raw-v1")
        en_ds = en_ds.map(lower_case_data)
        ds_to_json(en_ds, "en_ds_train.json")

        ger_ds = get_dataset("stefan-it/german-dbmdz-bert-corpus", split)
        ger_ds = ger_ds.map(lower_case_data)
        ds_to_json(ger_ds, "ger_ds_train.json")

        ff_en_ger = get_dataset("aari1995/false_friends_en_de", split)
        ff_en_ger = ff_en_ger.map(lower_case_data)
        ff_en_ger = ff_en_ger.rename_column("Sentence", "German Sentence")
        # Taking only subset of FF and filter out words that are not written exactly the same
        cols = ["False Friend", "Correct English Translation", "Wrong English Translation", "German Sentence"]
        ff_en_ger = ff_en_ger.select_columns(cols)
        ff_en_ger = ff_en_ger.filter(lambda row: ff_filter(row, "False Friend", "Wrong English Translation"))
        ff_en_ger = ds.Dataset.from_pandas(ff_en_ger.to_pandas().drop_duplicates(subset=["False Friend"]))
        with open("ff_eng_sentences.txt") as f:
            eng_sentences = f.read().splitlines()
        ff_en_ger = ff_en_ger.add_column(name="English Sentence", column=eng_sentences)
        ds_to_json(ff_en_ger, "ff_en_ger_train.json")
        
    

