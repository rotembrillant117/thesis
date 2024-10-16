# Promising false friends pairs:
# Note: From only starting the search for false friends language pairs, words that also sound the same but have a different meaning are also catagorized as false friends.
# English and Spanish have lots of false friends words that fall under that catagory, but, the words themselves seem to be spelled quite differently
# English-German https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_German
# English-French https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_French

from transformers import pipeline
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

def ds_to_csv(data, file_name):
  """
  Saves the Dataset to csv format
  :param data: the dataset
  :param file_name: the file name
  :return:
  """
  data.to_csv(file_name, index=False)


def lower_case_data(df):
  """
  Lower cases all the data in the dataset
  :param df: the pandas dataframe
  :return: the lowercased dataframe
  """
  for column in df.columns:
    df[column] = df[column].str.lower()
  return df



if __name__ == '__main__':

  unk_token = "<UNK>"  # token for unknown words
  spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens
  download = False
  # Loading and Saving the different datasets: english dataset, german dataset, false friends english-german
  if download:
    row_limit = 1000000
    split = f"train[:{row_limit}]"

    en_ds = get_dataset("Salesforce/wikitext", split, "wikitext-103-raw-v1")
    ds_to_csv(en_ds, "en_ds_train.csv")

    ger_ds = get_dataset("stefan-it/german-dbmdz-bert-corpus", split)
    ds_to_csv(ger_ds, "ger_ds_train.csv")

    ff_en_ger = get_dataset("aari1995/false_friends_en_de", split)
    ds_to_csv(ff_en_ger, "ff_en_ger_train.csv")

  # Loading files to pandas
  en_ds = pd.read_csv("en_ds_train.csv")
  ger_ds = pd.read_csv("ger_ds_train.csv")
  ff_en_ger = pd.read_csv("ff_en_ger_train.csv")

  # Taking only subset of FF df
  cols = ["False Friend", "Correct English Translation", "Wrong English Translation", "Sentence"]
  ff_en_ger = ff_en_ger[cols]

  # Lower casing all datasets
  en_ds = lower_case_data(en_ds)
  ger_ds = lower_case_data(ger_ds)
  ff_en_ger = lower_case_data(ff_en_ger)

  # False Friends dataset pre-processing: filter out words that are not written exactly the same
  mask = ff_en_ger["False Friend"] != ff_en_ger["Wrong English Translation"]
  ff_eng_ger = ff_en_ger[~mask].reset_index(drop=True)
  display(ff_eng_ger)


