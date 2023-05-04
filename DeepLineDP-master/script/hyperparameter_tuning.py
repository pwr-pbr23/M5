import os, re, argparse

from train_deep_line import train_model

arg = argparse.ArgumentParser()

arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')

args = arg.parse_args()

dataset_name = args.dataset

# constants from tuning perspective
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

def train_with_params(num_epochs, embed_dim, hidden_dim, num_layers, dropout, lr):
    word_gru_hidden_dim = hidden_dim
    print(f'setting word_gru_hidden_dim to {word_gru_hidden_dim}')
    sent_gru_hidden_dim = hidden_dim
    print(f'setting sent_gru_hidden_dim to {sent_gru_hidden_dim}')

    word_gru_num_layers = int(num_layers * num_layers_step)
    print(f'setting word_gru_num_layers to {num_layers}')
    sent_gru_num_layers = int(num_layers * num_layers_step)
    print(f'setting sent_gru_num_layers to {sent_gru_num_layers}')
    
    actual_batch_size = batch_size
    succeded = False

    exp_name = f'lr={lr}&dropout={dropout}&embed_dim={embed_dim}&hidden_dim={hidden_dim}&num_layers={num_layers}'
    while not succeded:
        print('training model: ', exp_name)
        try:
            train_model(dataset_name, actual_batch_size, num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr, exp_name)
            succeded = True
        except:
            actual_batch_size = int(batch_size / 2)
        

# may have higher value
num_epochs = 10

# determines the length of the word embedding vectors. 
# Typically, a higher embed_dim value results in word embeddings with more expressive power, but also requires more computational resources and may lead to overfitting
# Preprocesed data have vector of length 50
embed_dim = 50

def tune_model(
    lr_start, lr_count, lr_step, 
    dropout_start, dropout_count, dropout_step,
    num_layers_start, num_layers_count, num_layers_step, 
    hidden_dim_start, hidden_dim_count, hidden_dim_step):
    
    for num_layers in range(num_layers_start, num_layers_start + num_layers_count):
        num_layers = int(num_layers * num_layers_step)

        for hidden_dim in range(hidden_dim_start, hidden_dim_start + hidden_dim_count):
            hidden_dim = int(hidden_dim * hidden_dim_step)

            for dropout in range(dropout_start, dropout_start + dropout_count):
                dropout = dropout * dropout_step
                print(f'setting dropout to {dropout}')

                for lr in range(lr_start, lr_start + lr_count):
                    lr = lr * lr_step
                    print(f'setting learning rate to {lr}')

                    train_with_params(num_epochs, embed_dim, hidden_dim, num_layers, dropout, lr)


# learning rate is the step by which the weights in the model change
# learning rate in range 0.005 to 0.003 
lr_start = 1
lr_count = 6
lr_step = 0.0005

# dropout is the percentage of neurons that are zeroed in successive layers. It is to prevent overtraining
# dropout only useful when num_layers > 1
# dropout in range 0.05 to 0.03 
dropout_start = 1
dropout_count = 6
dropout_step = 0.05

# The Hierarchical Attention Network consists of two levels of attention mechanisms to capture the importance of different parts of the text.
    # word_gru_num_layers 
        # refers to the number of layers in the gated recurrent unit (GRU) network used to process the word-level inputs
        # In the HAN architecture, the word-level GRU processes the embeddings of individual words in the input text sequence. 
        # By stacking multiple GRU layers, the network can learn increasingly complex representations of the words and capture more nuanced relationships between them.
    # sent_gru_num_layers
        # refers to the number of layers in the gated recurrent unit (GRU) network used to process the sentence-level inputs.
        # Increasing the number of layers can improve the model's ability to capture complex patterns in the input text, but it can also make the model more difficult to train and more prone to overfitting.
# num layers in range 1 to 2
num_layers_start = 1
num_layers_count = 2
num_layers_step = 1

# The Hierarchical Attention Network consists of two levels of attention mechanisms to capture the importance of different parts of the text.
    # can potentially allow the model to learn more complex representations of the input text, but it can also increase the computational cost of training and may make the model more prone to overfitting.
    # word_gru_hidden_dim
        # refers to the number of hidden units in the gated recurrent unit (GRU) network used to process the word-level inputs.
    # sent_gru_hidden_dim
        # refers to the number of hidden units in the gated recurrent unit (GRU) network used to process the sentence-level inputs.
# hidden dim in range 32 to 128
hidden_dim_start = 1
hidden_dim_count = 4
hidden_dim_step = 32

tune_model(
    lr_start, lr_count, lr_step, 
    dropout_start, dropout_count, dropout_step, 
    num_layers_start, num_layers_count, num_layers_step,
    hidden_dim_start, hidden_dim_count, hidden_dim_step)