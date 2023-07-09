import json
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intens.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intents['patterns']:
        w = tokenize[pattern]
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', ',', '.']
al_words = [stem(w) for x in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(tag)

    label = tags.index(tag)
    Y_train.append(label)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batch_size = 0
    
dataset = ChatDataset()

train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
