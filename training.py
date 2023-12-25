from dataset_extraction import get_training_data, get_x_y
from speechbrain.nnet.RNN import LiGRU_Layer, SLiGRU_Layer
import torch
import torch.nn as nn
import random

class BuckeFillingRNNTModel(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer, out_size, batch_size=4) -> None:
        super().__init__()

        self.rnn = SLiGRU_Layer(input_size, hidden_size, n_layer, batch_size, dropout=0.2)
        self.decoder = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = self.rnn(x)
        # out, out_lens  = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.decoder(out)
        # out = nn.utils.rnn.pack_padded_sequence(out, out_lens, batch_first=True, enforce_sorted=False)
        return out

def create_batches(data_pairs, batch_size=4):
    batch_pairs = []
    x_feature_size = data_pairs[0][0].shape[0]
    y_feature_size = data_pairs[0][1].shape[0]
    max_len = max(data_pairs, key=lambda x: x[0].shape[1])[0].shape[1]
    x_batch_place_holder = torch.zeros((batch_size, max_len, x_feature_size))
    y_batch_place_holder = torch.zeros((batch_size, max_len, y_feature_size))
    x_input_lengths = []
    for i in range(len(data_pairs) // batch_size):
        x_input_lengths.append([])
        y_input_lengths = []
        x_sample = torch.clone(x_batch_place_holder)
        y_sample = torch.clone(y_batch_place_holder)
        samples = data_pairs[i * batch_size : (i+1) * batch_size]
        for i, (x, y) in enumerate(samples):
            l = x.shape[1]
            x_input_lengths[-1].append(l)
            y_input_lengths.append(l)
            x_sample[i, :l, :] = torch.from_numpy(x).T
            y_sample[i, :l, :] = torch.from_numpy(y).T
        # x_sample_ = nn.utils.rnn.pack_padded_sequence(x_sample, lengths=x_input_lengths, batch_first=True, enforce_sorted=False)
        # y_sample_ = nn.utils.rnn.pack_padded_sequence(y_sample, lengths=y_input_lengths, batch_first=True, enforce_sorted=False)
        batch_pairs.append((x_sample, y_sample))
    
    return batch_pairs, x_input_lengths

def zero_out(output, lengths):
    for i, l in enumerate(lengths):
        output[i, l:, :] = 0 * output[i, l:, :]

def train(batch_size=2, epoch_number=4):
    segments = get_training_data()
    data_pairs = []
    for segment in segments:
        x, y = get_x_y(segment)
        data_pairs.append((x,y))
    

    model = BuckeFillingRNNTModel(input_size=4, hidden_size=256, n_layer=2, out_size=3, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.3)
    for epoch in range(epoch_number):
        random.shuffle(data_pairs)
        batch_pairs, input_lengths = create_batches(data_pairs, batch_size=batch_size)
        for i, (x, y) in enumerate(batch_pairs):
            optimizer.zero_grad()
            output = model(x)
            zero_out(output, input_lengths[i])
            loss = criterion(output, y)
            print(f"loss: {loss} in epoch: {epoch}, iteration: {i}")
            loss.backward()
            optimizer.step()
    
    return model

    

if __name__ == "__main__":
    # segments = get_training_data()
    # data_pairs = []
    # for segment in segments:
    #     x, y = get_x_y(segment)
    #     data_pairs.append((x,y))
    
    train(batch_size=24, epoch_number=40)
    
