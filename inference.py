import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import random
import string
from collections import Counter

from nltk.corpus import stopwords
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, set_seed

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import warnings
warnings.filterwarnings("ignore")



def seed_everything(seed = 30):
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything()


class cfg:
    
    model = 'bert-base-uncased'
    seed = 16
    
    max_len = 512
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



data = pd.read_csv('/kaggle/input/stumbleupon/train.tsv', sep='\t')
test = pd.read_csv('/kaggle/input/stumbleupon/test.tsv',  sep='\t')

sub = pd.read_csv('/kaggle/input/stumbleupon/sampleSubmission.csv')



def lower_case(data):
    
    data['boilerplate'] = data['boilerplate'].str.lower()
    return data

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


cnt = Counter()
for text in data["boilerplate"].values:
    for word in text.split():
        cnt[word] += 1
        
FREQWORDS = set([w for (w, wc) in cnt.most_common(40)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

def preprocess_boilerplate(df):
    
    df['boilerplate'].replace(to_replace=r'"title":', value="", inplace=True, regex=True)
    df['boilerplate'].replace(to_replace=r'"url":'  , value="", inplace=True, regex=True)
    df['boilerplate'].replace(to_replace=r'"body":' , value="", inplace=True, regex=True)
    
    df = lower_case(df)
    
    df["boilerplate"] = df["boilerplate"].apply(lambda text: remove_punctuation(text))
    df["boilerplate"] = df["boilerplate"].apply(lambda text: remove_stopwords(text))
    df["boilerplate"] = df["boilerplate"].apply(lambda text: remove_freqwords(text))
    
    
    return df


data = preprocess_boilerplate(data)
test = preprocess_boilerplate(test)



def numerical_data_preprocessing(data):

    data['alchemy_category_score'] = pd.to_numeric(data['alchemy_category_score'], errors='coerce', downcast='float')
    data['alchemy_category_score'] = data['alchemy_category_score'].fillna(0.603) #mean of traiining data 
    
    data['is_news'] = pd.to_numeric(data['is_news'], errors='coerce', downcast='float')
    data['is_news'] = data['is_news'].fillna(0.0)
    
    data['news_front_page'] = pd.to_numeric(data['news_front_page'], errors='coerce', downcast='float')
    data['news_front_page'] = data['news_front_page'].fillna(0.5)

    return data

data = numerical_data_preprocessing(data)
test = numerical_data_preprocessing(test)



#Scaling the data

num_cols = list(data.columns[4:-1])

scaler = RobustScaler()
scaler.fit(data[num_cols])

data[num_cols] = scaler.transform(data[num_cols])
test[num_cols] = scaler.transform(test[num_cols])




train, valid = train_test_split(data, test_size = 0.2, random_state = cfg.seed, stratify = data['label'])
train, valid = train.reset_index(), valid.reset_index()



tokenizer = BertTokenizer.from_pretrained(cfg.model)

def tokenizing(data):
    
    
    encoded_data = tokenizer.batch_encode_plus(list(data.boilerplate.values), 
                                               add_special_tokens=True, 
                                               return_attention_mask=True, 
                                               #pad_to_max_length=True, 
                                               padding='max_length',
                                               max_length=cfg.max_len, 
                                               return_tensors='pt',
                                               truncation=True)
    
    return encoded_data['input_ids'], encoded_data['attention_mask']


def make_Tensor(data, test = False):
    
    
    ids , att = tokenizing(data)
    
    num = data[num_cols].values
    
    if test:
        return ids, att, torch.Tensor(num), -1
    
    y = data['label'].values
    
    return ids, att, torch.Tensor(num), torch.Tensor(y)
    

    
trn_ids, trn_att, trn_num, trn_y = make_Tensor(train)
vld_ids, vld_att, vld_num, vld_y = make_Tensor(valid)
t_ids, t_att, t_num, _ = make_Tensor(test)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
    
###########################################################################


trn_loader = FastTensorDataLoader(trn_ids, trn_att, trn_num, trn_y, batch_size = cfg.bs, shuffle = True)
vld_loader = FastTensorDataLoader(vld_ids, vld_att, vld_num, vld_y, batch_size = cfg.bs, shuffle = False)

t_loader = FastTensorDataLoader(t_ids, t_att, t_num, batch_size = cfg.bs, shuffle = False)


class MODEL(nn.Module):

    def __init__(self):
        super(MODEL, self).__init__()
        self.backbone =  BertForSequenceClassification.from_pretrained(cfg.model, num_labels=1)
        self.NumL = nn.Linear(in_features= 22,out_features=4)
        self.out  = nn.Linear(in_features= 5 ,out_features=1)

    def forward(self, input_ids, attention_masks, num):
        
        x = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        y = torch.nn.functional.relu(self.NumL(num))
        
        z = torch.cat((x['logits'], y), 1)
        z = self.out(z)
        
        return torch.nn.functional.sigmoid(z)

    
###################################################################################################
def train_func(model, data_loader, criterion, optimizer):
    train_loss = 0.0
    
    model.train()
    for ids, att, num, y in data_loader:
        
        ids = ids.to(device)
        att = att.to(device)
        num = num.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        y_preds = model(ids, att, num).squeeze()
        
        loss = criterion(y_preds, y)
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.005)
        
        optimizer.step()
        train_loss += loss.item()

        
    return train_loss / len(data_loader)
###################################################################################################

###################################################################################################    
def pred_func(model, data_loader):
    valid_preds =  []
    valid_loss = 0.0
    
    model.eval()
    for ids, att, num, y in data_loader:
        
        ids = ids.to(device)
        att = att.to(device)
        num = num.to(device)
        
        with torch.no_grad():
            y_preds = model(ids, att, num).squeeze()
        
        valid_preds.append(y_preds.to('cpu').detach().numpy())
        
    return valid_loss / len(data_loader), np.concatenate(valid_preds)
###################################################################################################




model = MODEL()                
model.to(device)

preds = pred_func(model, t_loader)

sub['label'] = preds
sub.to_csv('sub.csv', index = False)
