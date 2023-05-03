import os, re

import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm

from sklearn.utils import compute_class_weight

from DeepLineDP_model import *
from my_util import *

torch.manual_seed(0)

max_grad_norm = 5
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True

save_every_epochs = 1

max_train_LOC = 900

prediction_dir = '../output/prediction/DeepLineDP/'
save_model_dir = '../output/model/DeepLineDP/'

file_lvl_gt = '../datasets/preprocessed_data/'


def get_loss_weight(labels, weight_dict):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1,1).cuda()
    return weight_tensor

def train_model(dataset_name, batch_size, num_epochs, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, dropout, lr, exp_name = ''):
    weight_dict = {}

    loss_dir = '../output/loss/DeepLineDP/' + dataset_name + '/'
    actual_save_model_dir = save_model_dir+dataset_name+'/'

    if not exp_name == '':
        actual_save_model_dir = actual_save_model_dir+exp_name+'/'
        loss_dir = loss_dir + exp_name + '/'

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel)
    valid_df = get_df(valid_rel)

    train_code3d, train_label = get_code3d_and_label(train_df, True)
    valid_code3d, valid_label = get_code3d_and_label(valid_df, True)

    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)

    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)
    
    w2v_dir = get_w2v_path()

    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')

    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.vocab)  + 1 # for unknown tokens

    x_train_vec = get_x_vec(train_code3d, word2vec)
    x_valid_vec = get_x_vec(valid_code3d, word2vec)

    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)

    train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)

    valid_dl = get_dataloader(x_valid_vec, valid_label,batch_size,max_sent_len)

    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout)

    model = model.cuda()
    model.sent_attention.word_attention.freeze_embeddings(False)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCELoss()

    checkpoint_files = [file for file in os.listdir(actual_save_model_dir) if os.path.isfile(os.path.join(actual_save_model_dir, file))] 

    if '.ipynb_checkpoints' in checkpoint_files:
        checkpoint_files.remove('.ipynb_checkpoints')

    total_checkpoints = len(checkpoint_files)

    # no model is trained 
    if total_checkpoints == 0:
        model.sent_attention.word_attention.init_embeddings(word2vec_weights)
        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []
    
    else:
        print(f'total_checkpoints={total_checkpoints}')
        checkpoint_nums = [int(re.findall('\d+', re.findall('\d+epochs',s)[0])[0]) for s in checkpoint_files]
        current_checkpoint_num = max(checkpoint_nums)

        checkpoint_file_name = [re.sub('\d+epochs', f'{current_checkpoint_num}epochs', s) for s in checkpoint_files][0]

        checkpoint = torch.load(actual_save_model_dir+checkpoint_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        loss_df = pd.read_csv(loss_dir+dataset_name+'-loss_record.csv')
        train_loss_all_epochs = list(loss_df['train_loss'])
        val_loss_all_epochs = list(loss_df['valid_loss'])

        current_checkpoint_num = current_checkpoint_num+1 # go to next epoch
        print('continue training model from epoch',current_checkpoint_num)

    for epoch in tqdm(range(current_checkpoint_num,num_epochs+1)):
        train_losses = []
        val_losses = []

        model.train()

        for inputs, labels in train_dl:

            inputs_cuda, labels_cuda = inputs.cuda(), labels.cuda()
            output, _, __, ___ = model(inputs_cuda)

            weight_tensor = get_loss_weight(labels, weight_dict)

            criterion.weight = weight_tensor

            loss = criterion(output, labels_cuda.reshape(batch_size,1))

            train_losses.append(loss.item())
            
            torch.cuda.empty_cache()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            torch.cuda.empty_cache()

        train_loss_all_epochs.append(np.mean(train_losses))

        with torch.no_grad():
            
            criterion.weight = None
            model.eval()
            
            for inputs, labels in valid_dl:

                inputs, labels = inputs.cuda(), labels.cuda()
                output, _, __, ___ = model(inputs)
            
                val_loss = criterion(output, labels.reshape(batch_size,1))

                val_losses.append(val_loss.item())

            val_loss_all_epochs.append(np.mean(val_losses))

        if epoch % save_every_epochs == 0:
            print(dataset_name,'- at epoch:',str(epoch))

            if exp_name == '':
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+str(epoch)+'epochs.pth')
            else:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, 
                            actual_save_model_dir+'checkpoint_'+exp_name+'_'+str(epoch)+'epochs.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1,len(train_loss_all_epochs)+1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        
        loss_df.to_csv(loss_dir+dataset_name+'-loss_record.csv',index=False)