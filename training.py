from dataset_extraction import main
import torch
import torch.nn as nn

def get_model():
    rnn = nn.GRU(4, 20, 4)

def train():
    segments = main()
    
