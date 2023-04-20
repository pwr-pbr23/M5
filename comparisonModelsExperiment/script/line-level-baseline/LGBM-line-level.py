from lightgbm import LGBMClassifier

import os, sys, pickle

import numpy as np

from gensim.models import Word2Vec
from torch.utils import data

from tqdm import tqdm

sys.path.append('../')
from my_util import *

model_dir = '../../output/model/LGBM-line-level/'
result_dir = '../../output/LGBM-line-level-result/'

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

def train_and_predict(dataset_name):
    word2vec = get_W2V(dataset_name)

    train_rel = all_train_releases[dataset_name]

    train_df = get_df(train_rel, is_baseline=True)
    
    line_rep_list = []
    all_line_label = []

    for _, df in tqdm(train_df.groupby('filename')):

        code = df['code_line'].tolist()
        line_label = df['line-label'].tolist()

        all_line_label.extend(line_label)
    
        code2d = prepare_code2d(code, to_lowercase)
        code3d = [code2d]
        codevec = get_x_vec(code3d, word2vec)

        simplified_code_vec = list(map(lambda v: v[0:50], codevec[0]))

        line_rep_list.append(simplified_code_vec)

    X_train = np.concatenate(line_rep_list)
    y_train = all_line_label


    clf = LGBMClassifier(n_estimators=100, base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1,n_jobs=24)

    clf.fit(X_train, y_train)

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
            X_test = list(map(lambda v: v[0:50], codevec[0]))

            y_pred = clf.predict(X_test) # true or false

            print(X_test, y_pred)
            df['line-score-pred'] = y_pred.astype(int) # 1 or 0, if 1 then the label is correct (ie. l should have no defect and is labeled as no defect), else 0


            all_df_list.append(df)

        all_df = pd.concat(all_df_list)

        all_df.to_csv(result_dir+rel+'-line-lvl-result.csv',index=False)

        print('finished',rel)


proj_name = sys.argv[1]

train_and_predict(proj_name)