import os
import tqdm
import torch
import pickle
import sacremoses
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader, Dataset
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import average_precision_score

BATCH_SIZE = 64
max_sent_length=128
hidden_size = 128
num_layers = 2
num_classes = 2
bidirectional=True
torch.manual_seed(1234)
early_stop_patience=3
NUM_EPOCHS=10
have_pickle_file = True
spam_train_name = '/fake_news/'
tokens_save_dir = '/scratch/xl3119/tokenized_data.p'
model_save_dir = '/scratch/xl3119/best_model_LSTM_max_pooling_no_sample.pt'
over_sample = False
under_sample = False
model_type = 'LSTM'
## Load glove embedding
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
EMBEDDING_DIM=300 # dimension of Glove embeddings
glove_path = "/scratch/xl3119/glove.6B.300d__50k.txt"

## Define Fake News Class
class Fake_News_Dataset(Dataset):

    def __init__(self, data_list, target_list, max_sent_length=128):
        self.data_list = data_list
        self.target_list = target_list
        self.max_sent_length = max_sent_length
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key, max_sent_length=None):
        if max_sent_length is None:
            max_sent_length = self.max_sent_length
        token_idx = self.data_list[key][:max_sent_length]
        label = self.target_list[key]
        return [token_idx, label]

    def spam_collate_func(self, batch):
        data_list = [] # store padded sequences
        label_list = []
        max_batch_seq_len = max([len(datum[0]) for datum in batch])
        if max_batch_seq_len >= self.max_sent_length:
            max_batch_seq_len = self.max_sent_length
        for datum in batch:
            label_list.append(datum[1])
            data_list.append(datum[0][:max_batch_seq_len]+[0]*(max(0,max_batch_seq_len - len(datum[0]))))

        return [torch.LongTensor(data_list), torch.LongTensor(label_list)]

## Over and Undersample
def oversample(X_train, y_train):
    rus = RandomUnderSampler()
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    return X_train_rus, y_train_rus

def undersample(X_train, y_train):
    ros = RandomOverSampler()
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

## MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers, num_classes, bidirectional, dropout_prob=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout_prob)
        self.non_linearity = nn.ReLU()
        self.embedding_layer = self.load_pretrained_embeddings(embeddings)
        self.l1 = nn.Linear(embeddings.shape[1], 128)
        self.l2 = nn.Linear(128, 128)
        self.clf = nn.Linear(128, num_classes)

    def load_pretrained_embeddings(self, embeddings):
        embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        embedding_layer.weight.data = torch.Tensor(embeddings).float()
        return embedding_layer

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        AvgPool1d = nn.AvgPool1d(x.size(1), stride = 1)
        x = AvgPool1d(torch.transpose(x, 1, 2)).sum(-1)
        x = self.l1(x)
        x = self.non_linearity(x)
        x = self.l2(x)
        x = self.non_linearity(x)
        logits = self.clf(x)

        return logits

## LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers, num_classes, bidirectional, dropout_prob=0.3):
        super().__init__()
        self.embedding_layer = self.load_pretrained_embeddings(embeddings)
        self.dropout = nn.Dropout(p = dropout_prob)
        self.lstm = nn.LSTM(embeddings.shape[1],
                            hidden_size,
                            num_layers,
                            dropout = dropout_prob,
                            bidirectional = bidirectional,
                            batch_first=True)
        self.non_linearity = nn.ReLU() # For example, ReLU
        self.clf = nn.Linear((1+int(bidirectional))*hidden_size, num_classes) # classifier layer


    def load_pretrained_embeddings(self, embeddings):
        embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
        embedding_layer.weight.data = torch.Tensor(embeddings).float()
        return embedding_layer


    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        x, _ = self.lstm(x)
        MaxPool1d = nn.MaxPool1d(x.size(1), stride = 1)
        x = MaxPool1d(torch.transpose(x, 1, 2)).sum(-1)
        x = self.non_linearity(x)
        logits = self.clf(x)

        return logits

def load_glove(glove_path, embedding_dim):
    with open(glove_path) as f:
        token_ls = [PAD_TOKEN, UNK_TOKEN]
        embedding_ls = [np.zeros(embedding_dim), np.random.rand(embedding_dim)]
        for line in f:
            token, raw_embedding = line.split(maxsplit=1)
            token_ls.append(token)
            embedding = np.array([float(x) for x in raw_embedding.split()])
            embedding_ls.append(embedding)
        embeddings = np.array(embedding_ls)
    return token_ls, embeddings

def featurize(data, labels, tokenizer, vocab, max_seq_length=128):
    vocab_to_idx = {word: i for i, word in enumerate(vocab)}
    text_data = []
    label_data = []
    for ex in tqdm.tqdm(data):
        tokenized = tokenizer.tokenize(ex.lower())
        ids = [vocab_to_idx.get(token, 1) for token in tokenized]
        text_data.append(ids)
    return text_data, labels

## Evaluate data
def evaluate(model, dataloader, device):
    accuracy = None
    scores = []
    labels = []

    model.eval()
    with torch.no_grad():
        num_true_pred = 0
        num_pred = 0
        for batch_text, batch_labels in dataloader:
            #Get scores and labels
            sigmoid = nn.Sigmoid().cuda()
            scores.extend(sigmoid(model(batch_text.to(device)))[:,1].tolist())
            labels.extend(batch_labels.tolist())
            #Get all predicted labels
            preds = ((model(batch_text.to(device)) >= 0.5)[:,1].long() == batch_labels.to(device)).long()
            #Compute accuracies
            num_true_pred += preds.sum(0).item()
            num_pred += preds.size(0)
    scores = np.array(scores)
    labels = np.array(labels)
    accuracy = num_true_pred / num_pred
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return accuracy, auc, ap

vocab, embeddings = load_glove(glove_path, EMBEDDING_DIM)

if have_pickle_file:

    pickle_fake_news = pickle.load(open(tokens_save_dir, "rb"))
    train_data_indices = pickle_fake_news['train_indices']
    train_labels = pickle_fake_news['train_labels']
    val_data_indices = pickle_fake_news['val_indices']
    val_labels = pickle_fake_news['val_labels']

    assert(not (over_sample and under_sample))
    if over_sample or under_sample:
            data_idx = np.array([[i] for i in range(len(train_labels))])
            if over_sample:
                new_data_idx, train_labels = oversample(data_idx, np.array(train_labels))
            else:
                new_data_idx, train_labels = undersample(data_idx, np.array(train_labels))
            train_data_indices = [train_data_indices[elem] for elem in new_data_idx.transpose().tolist()[0]]
            train_labels = train_labels.tolist()

else:
    ## Load training data
    span_train_name = ''
    train_df = pd.read_csv('/scratch/xl3119/'+spam_train_name+'train.csv')
    val_df = pd.read_csv('/scratch/xl3119/'+spam_train_name+'dev.csv')

    train_texts, train_labels, train_rating = list(train_df.review), list(train_df.label), list(train_df.rating)
    val_texts, val_labels, val_rating     = list(val_df.review), list(val_df.label), list(val_df.rating)

    print(
        f"Train size: {len(train_labels)}\n"
        f"Val size: {len(val_labels)}\n"
    )

    ## Tokenize data
    tokenizer = sacremoses.MosesTokenizer()
    train_data_indices, train_labels = featurize(train_texts, train_labels, tokenizer, vocab)
    val_data_indices, val_labels = featurize(val_texts, val_labels, tokenizer, vocab)

    pickle_fake_news = {'train_indices': train_data_indices,
                        'train_labels': train_labels,
                        'train_rating': train_rating,
                        'val_indices': val_data_indices,
                        'val_labels': val_labels,
                        'val_rating': val_rating,}

    pickle.dump(pickle_fake_news,open(tokens_save_dir, "wb"))
    print('Data has been saved')

## Build data loader
train_dataset = Fake_News_Dataset(train_data_indices, train_labels, max_sent_length)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.spam_collate_func,
                                           shuffle=True)

val_dataset = Fake_News_Dataset(val_data_indices, val_labels, train_dataset.max_sent_length)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.spam_collate_func,
                                           shuffle=False)

## Check if GPU is available
print('Cuda availability: {}'.format(str(torch.cuda.is_available())))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device=torch.device('cpu')

## Setup model

if model_type == 'LSTM':
    model = LSTMClassifier(embeddings, hidden_size, num_layers, num_classes, bidirectional)
elif model_type == 'MLP':
    model = MLPClassifier(embeddings, hidden_size, num_layers, num_classes, bidirectional)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

## Begin training
train_loss_history = []
val_accuracy_history = []
val_auc_history = []
val_ap_history = []
best_val_accuracy = 0
best_val_auc = 0
best_val_ap = 0
n_no_improve = 0

for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
    model.train() # this enables regularization, which we don't currently have
    for i, (data_batch, batch_labels) in tqdm.tqdm(enumerate(train_loader)):
        preds = model(data_batch.to(device))
        loss = criterion(preds, batch_labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_history.append(loss.item())

    # The end of a training epoch
    temp_val_acc, temp_val_auc, temp_val_ap = evaluate(model, val_loader, device)
    val_accuracy_history.append(temp_val_acc)
    val_auc_history.append(temp_val_auc)
    val_ap_history.append(temp_val_ap)
    print('epoch:{}, acc:{}, auc:{}, ap:{}'.format(str(epoch+1),
                                                    str(temp_val_acc),
                                                    str(temp_val_auc),
                                                    str(temp_val_ap),))

    if temp_val_ap > best_val_ap:
        best_val_ap = temp_val_ap
        torch.save(model, model_save_dir)
    else:
        n_no_improve += 1
        if n_no_improve > early_stop_patience: break
    print('Evaluation accuracy after epoch {}: {}'.format(str(i+1), temp_val_acc))

print("Best validation accuracy is: ", best_val_ap)
