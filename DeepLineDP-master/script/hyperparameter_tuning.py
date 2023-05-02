import os, re, argparse

from train_deep_line import train_model

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')

args = arg.parse_args()

dataset_name = args.dataset

batch_size = 8
num_epochs = 10
max_grad_norm = 5
embed_dim = 50
word_gru_hidden_dim = 64
sent_gru_hidden_dim = 64
word_gru_num_layers = 1
sent_gru_num_layers = 1
use_layer_norm = True
dropout = 0.2
lr = 0.001

save_every_epochs = 1
exp_name = ''

max_train_LOC = 900

prediction_dir = '../output/prediction/DeepLineDP/'
save_model_dir = '../output/model/DeepLineDP/'

file_lvl_gt = '../datasets/preprocessed_data/'

weight_dict = {}

train_model(dataset_name, batch_size, num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr, exp_name)