from os.path import expanduser
import nltk

from torchtext import data, vocab
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator, Pipeline

import torch
import torch.nn as nn

import numpy as np

from sklearn.model_selection import GroupKFold

from cnn_model import CNNFeatureExtractor, UnMaskedWeightedNLLLoss

import torch.optim as optim

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support
import time

import pickle

def train_or_eval_model(model, dataloader, loss_function=None, optimizer=None, train=False, valid=False, test=False):
    losses = []
    preds = []
    labels = []
    features = []
    ids = []
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        #print(data.text.size())
        features_, log_prob = model(data) # batch, n_classes
        features.append(features_)
        ids.append(data.id)
        lp_ = log_prob # batch, n_classes
        # import ipdb;ipdb.set_trace()
        if train or valid or test:
            labels_ = data.label # batch
            #print(lp_.size())
            #print(labels_.size())
            loss = loss_function(lp_, labels_)
            losses.append(loss.item())
            labels.append(labels_.data.cpu().numpy())

        pred_ = torch.argmax(lp_,1) # batch
        preds.append(pred_.data.cpu().numpy())

        if train:
            loss.backward()
            # if args.tensorboard:
            #     for param in model.named_parameters():
            #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if train or valid or test:
        if preds!=[]:
            # import ipdb;ipdb.set_trace()
            preds  = np.concatenate(preds)
            labels = np.concatenate(labels)
        else:
            return float('nan'), float('nan'), [], [], float('nan')

        avg_loss = round(np.sum(losses)/len(labels),4)
        avg_accuracy = round(accuracy_score(labels,preds)*100,2)
        #_,_,_,avg_fscore = get_metrics(labels,preds)
        avg_fscore = round(f1_score(labels,preds,average='macro')*100,2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore, features, ids
    else:
        preds  = np.concatenate(preds)
        return preds, features, ids
id    = data.Field(sequential=False)
label = data.LabelField()
sent  = data.Field(tokenize=nltk.word_tokenize, lower=True)

fields = [('id', id),
          ('text', sent),
          ('label', label),
          ]


dataset = data.TabularDataset('ddt/dailydialog.csv',
                            format='csv',
                            fields=fields,
                            )
label.build_vocab(dataset)

vectors = vocab.Vectors(name='glove.840B.300d.txt', cache='content/drive/')

id.build_vocab(dataset)
label.build_vocab(dataset)
sent.build_vocab(dataset, vectors=vectors)
#[protects the non-nested fields(.py#629) from flattening ]

embedding_vectors = sent.vocab.vectors
vocab_size = len(sent.vocab)

train = np.array(dataset.examples[:87170])
valid = np.array(dataset.examples[87170:95239])
test  = np.array(dataset.examples[95239:])

train_ds = data.Dataset(train, fields)
valid_ds = data.Dataset(valid, fields)
test_ds  = data.Dataset(test, fields)

train_loader = BucketIterator(train_ds,
                            train=True,
                            batch_size=200,
                            shuffle = True,
                            # sort_key=lambda x: x.id,
                            # device=torch.device(0),
                            )

valid_loader = BucketIterator(valid_ds,
                            batch_size=200,
                            shuffle = True,
                            # sort_key=lambda x: x.id,
                            # device=torch.device(0),
                            )

test_loader = BucketIterator(test_ds,
                           batch_size=200,
                           shuffle = True,
                           # sort_key=lambda x: x.id,
                           # device=torch.device(0),
                           )



loss_function = UnMaskedWeightedNLLLoss()

embedding_dim = 300
output_size = 100
filters = 50
kernel_sizes = [3,4,5]
dropout = 0.5
model = CNNFeatureExtractor(embedding_vectors, output_size, filters, kernel_sizes, dropout)
optimizer = optim.Adam(model.parameters())
n_epochs = 30
for e in range(n_epochs):
      start_time = time.time()
      train_loss, train_acc, _,_,train_fscore, features_train, ids_train = train_or_eval_model(model,
                                               train_loader, loss_function, optimizer, train=True)
      valid_loss, valid_acc, valid_label, valid_pred, val_fscore, features_valid, ids_valid = train_or_eval_model(model, valid_loader, loss_function, valid=True)
      test_loss, test_acc, test_label, test_pred, test_fscore, features_test, ids_test = train_or_eval_model(model, test_loader, loss_function, test=True)
      print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore, test_loss, test_acc, test_fscore, \
                        round(time.time()-start_time,2)))
      with open(str(e+1)+'_train_features.p','wb') as f:
        pickle.dump([ids_train,features_train],f)
      with open(str(e+1)+'_valid_features.p','wb') as f:
        pickle.dump([ids_valid,features_valid],f)

      with open(str(e+1)+'_test_features.p','wb') as f:
        pickle.dump([ids_test,features_test],f)
#_,features_train, ids_train=train_or_eval_model(model, train_loader,loss_function=loss_function)
#_,features_valid, ids_valid=train_or_eval_model(model, valid_loader,loss_function=loss_function)
#_,features_test, ids_test=train_or_eval_model(model, test_loader,loss_function=loss_function)

#print(features_test)

