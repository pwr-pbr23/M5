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

def train_with_params(num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, num_layers, dropout, lr):
    word_gru_num_layers = int(num_layers * num_layers_step)
    print(f'setting word_gru_num_layers to {num_layers}')
    sent_gru_num_layers = int(num_layers * num_layers_step)
    print(f'setting sent_gru_num_layers to {sent_gru_num_layers}')
    
    actual_batch_size = calculate_batch_size(num_layers)
    print(f'setting batch_size to {actual_batch_size}')

    exp_name = f'lr={lr}&dropout={dropout}&num_layers={num_layers}'
    print('training model: ', exp_name)
    train_model(dataset_name, actual_batch_size, num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr, exp_name)

def tune_model(lr_start, lr_count, lr_step, dropout_start, dropout_count, dropout_step, num_layers_start, num_layers_count, num_layers_step):
    for lr in range(lr_start, lr_start + lr_count):
        lr = lr * lr_step
        print(f'setting learning rate to {lr}')

        for dropout in range(dropout_start, dropout_start + dropout_count):
            dropout = dropout * dropout_step
            print(f'setting dropout to {dropout}')

            for num_layers in range(num_layers_start, num_layers_start + num_layers_count):
                num_layers = int(num_layers * num_layers_step)
                
                train_with_params(num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, num_layers, dropout, lr)

# learning rate in range 0.005 to 0.003 
lr_start = 1
lr_count = 6
lr_step = 0.0005

# dropout in range 0.05 to 0.03 
dropout_start = 1
dropout_count = 6
dropout_step = 0.05

# num layers in range 1 to 2
num_layers_start = 1
num_layers_count = 2
num_layers_step = 1

tune_model(lr_start, lr_count, lr_step, dropout_start, dropout_count, dropout_step, num_layers_start, num_layers_count, num_layers_step)