import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, word_embeddings, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1])
        self.embedding.weight.data.copy_(word_embeddings)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=word_embeddings.shape[1], out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size
        self.smax_fc = nn.Linear(output_size, 7)

    def forward(self, x):
        
        #batch, num_words = x.size()
        
        #x = x.type(LongTensor)  # (batch, num_words)
        text = x.text.transpose(1,0)
        emb = self.embedding(text) # (batch, num_words) -> (batch, num_words, 300) 
        emb = emb.transpose(-2, -1).contiguous() # (batch, num_words, 300)  -> (batch, 300, num_words) 
        
        convoluted = [F.relu(conv(emb)) for conv in self.convs] 
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] 
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (batch, 150) -> (batch, 100)
        log_prob = F.log_softmax(self.smax_fc(features), -1)
        return features, log_prob
class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss