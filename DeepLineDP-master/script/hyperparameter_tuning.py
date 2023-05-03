import os, re, argparse

from train_deep_line import train_model

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')

args = arg.parse_args()

dataset_name = args.dataset

# describe
# lr = 0.001                      # krok w jaki sposób się zmieniają wagi
# word_gru_num_layers = 1
# sent_gru_num_layers = 1
# dropout only useful when num_layers > 1
# dropout = 0.2                   # wyższe wartości - więcej zerowanych neuronów

# give try
num_epochs = 10                 # można spróbować wyższe

word_gru_hidden_dim = 64
sent_gru_hidden_dim = 64


embed_dim = 50                  # większa wartość może poprawić, ale kosztowna

# not
batch_size = 8
use_layer_norm = True
max_grad_norm = 5

save_every_epochs = 1

def calculate_batch_size(num_layers):
    # CPU requirements
    if num_layers > 1:
        return int(batch_size / 2)
    else:
        return batch_size

def train_with_params(num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr):
    actual_batch_size = calculate_batch_size(num_layers)
    print(f'setting batch_size to {actual_batch_size}')

    exp_name = f'lr={lr}&dropout={dropout}&num_layers={num_layers}'
    print('training model: ', exp_name)
    train_model(dataset_name, actual_batch_size, num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr, exp_name)

#learning rate in range 0.005 to 0.003 
for lr in range(1, 7):
    lr = lr * 0.0005
    print(f'setting learning rate to {lr}')

    #dropout in range 0.05 to 0.03 
    for dropout in range(1,7):
        dropout = dropout * 0.05
        print(f'setting dropout to {dropout}')

        for num_layers in range(1, 3):
            word_gru_num_layers = num_layers
            print(f'setting word_gru_num_layers to {num_layers}')
            sent_gru_num_layers = num_layers
            print(f'setting sent_gru_num_layers to {sent_gru_num_layers}')

            train_with_params(num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr)