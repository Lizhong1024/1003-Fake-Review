import numpy as np
import pandas as pd
import spacy
import pickle as pkl

def tokenize(data):
    
    nlp = spacy.load('en_core_web_sm')
    data_tokenized = []

    def tokenize_(review):
        doc = nlp(review)
        tokens = []
        for tk in doc:
            if tk.is_punct or tk.is_stop or tk.is_space:
                continue # discard if is punctuation / stopword / whitespace
            elif any([char.isdigit() for char in tk.text]):
                continue # discard if is a number
            else:
                tokens.append(tk.lemma_.lower())
        return tokens
    
    for review in data:
        tokens = tokenize_(review)
        data_tokenized.append(tokens)

    return data_tokenized


def get_vocab_set(data_tokenized):
    return set.union(*[set(tokens) for tokens in data_tokenized])


def main(train_path, val_path, test_path):
    # Load data
    train = pd.read_csv(train_path, index_col='ex_id')
    dev = pd.read_csv(val_path, index_col='ex_id')
    test = pd.read_csv(test_path, index_col='ex_id')

    # Tain
    print ("Tokenizing train data")
    train_data_tokens = tokenize(train.review.values)
    pkl.dump(train_data_tokens, open("data/train_data_tokens.pkl", "wb"))

    # Val
    print ("Tokenizing val data")
    val_data_tokens = tokenize(dev.review.values)
    pkl.dump(val_data_tokens, open("data/val_data_tokens.pkl", "wb"))

    # Test
    print ("Tokenizing test data")
    test_data_tokens = tokenize(test.review.values)
    pkl.dump(test_data_tokens, open("data/test_data_tokens.pkl", "wb"))

    # Vocab
    print("Creating vocab set")
    all_train_tokens = get_vocab_set(train_data_tokens)
    pkl.dump(all_train_tokens, open("data/all_train_tokens.pkl", "wb"))

    print("Finished")


if __name__ == '__main__':

    train_path = 'data/train.csv'
    val_path = 'data/dev.csv'
    test_path = 'data/test_no_label.csv'

    main(train_path, val_path, test_path)