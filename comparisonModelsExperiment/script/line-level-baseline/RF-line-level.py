from sklearn.ensemble import RandomForestClassifier

import os, sys, pickle

import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from torch.utils import data

from tqdm import tqdm

sys.path.append('../')
from my_util import *

model_dir = '../../output/model/RF-line-level/'
result_dir = '../../output/RF-line-level-result/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

embed_dim = 50
to_lowercase = True

def get_W2V(dataset_name):
    w2v_dir = get_w2v_path()
    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')
    word2vec = Word2Vec.load('../'+word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')
    return word2vec

def train_RF_model(dataset_name):
    word2vec = get_W2V(dataset_name)

    clf = RandomForestClassifier(random_state=0, n_jobs=24)

    train_rel = all_train_releases[dataset_name]

    train_df = get_df(train_rel, is_baseline=True)
    
    line_rep_list = []
    all_line_label = []

    # loop to get line representation of each file in train data
    for _, df in tqdm(train_df.groupby('filename')):

        code = df['code_line'].tolist()
        line_label = df['line-label'].tolist()

        all_line_label.extend(line_label)
    
        code2d = prepare_code2d(code, to_lowercase)
        code3d = [code2d]
        codevec = get_x_vec(code3d, word2vec)

        simplified_code_vec = list(map(lambda v: v[0:50], codevec[0]))

        line_rep_list.append(simplified_code_vec)

    x = np.concatenate(line_rep_list)

    print('prepare data finished', len(x))

    # global y_train
    # y_train = all_line_label
    clf.fit(x,all_line_label)

    pickle.dump(clf, open(model_dir+dataset_name+'-RF-model.bin','wb'))

    print('finished training model of',dataset_name)


def predict_defective_line(dataset_name):
    word2vec = get_W2V(dataset_name)
    clf = pickle.load(open(model_dir+dataset_name+'-RF-model.bin','rb'))

    print('load model finished')

    test_rels = all_eval_releases[dataset_name][1:]

    for rel in test_rels:
        test_df = get_df(rel, is_baseline=True)

        test_df = test_df[test_df['file-label']==True]
        test_df = test_df.drop(['is_comment','is_test_file','is_blank'],axis=1)

        all_df_list = [] # store df for saving later...

        for _, df in tqdm(test_df.groupby('filename')):

            code = df['code_line'].tolist()

            code2d = prepare_code2d(code, to_lowercase)

            code3d = [code2d]

            codevec = get_x_vec(code3d, word2vec)
            simplified_code_vec = list(map(lambda v: v[0:50], codevec[0]))

            pred = clf.predict(simplified_code_vec)

            df['line-score-pred'] = pred.astype(int) # 1 or 0, if 1 then the label is correct (ie. l should have no defect and is labeled as no defect), else 0

            all_df_list.append(df)

        all_df = pd.concat(all_df_list)

        all_df.to_csv(result_dir+rel+'-line-lvl-result.csv',index=False)

        print('finished',rel)

proj_name = sys.argv[1]

train_RF_model(proj_name)
predict_defective_line(proj_name)
