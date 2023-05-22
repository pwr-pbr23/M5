from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import os, sys
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import RandomizedSearchCV
from torch.utils import data
from tqdm import tqdm

sys.path.append('../')
from my_util import *

embed_dim = 50
to_lowercase = True

def get_W2V(dataset_name):
    w2v_dir = get_w2v_path()
    word2vec_file_dir = os.path.join(w2v_dir,dataset_name+'-'+str(embed_dim)+'dim.bin')
    word2vec = Word2Vec.load('../'+word2vec_file_dir)
    print('load Word2Vec for',dataset_name,'finished')
    return word2vec

def get_XY_train(dataset_name, word2vec):
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

    return X_train, y_train

def train_and_predict(dataset_name, classifierName, Classifier, params):
    model_dir = f'../../output/model/{classifierName}-line-level/'
    result_dir = f'../../output/{classifierName}-line-level-result/'

    for path in [model_dir, result_dir]:
        if not os.path.exists(path):    
            os.makedirs(path) 

    word2vec = get_W2V(dataset_name)

    X_train, y_train = get_XY_train(dataset_name, word2vec)    
    
    seed = np.random.seed(0)

    clf = Classifier(n_jobs=24, **params, random_state=seed)

    clf.fit(X_train, y_train)

    test_rels = all_eval_releases[dataset_name][1:]

    for rel in test_rels:
        test_df = get_df(rel, is_baseline=True)

        test_df = test_df[test_df['file-label']==True]
        test_df = test_df.drop(['is_comment','is_test_file','is_blank'],axis=1)

        all_df_list = []

        for _, df in tqdm(test_df.groupby('filename')):

            code = df['code_line'].tolist()

            code2d = prepare_code2d(code, to_lowercase)

            code3d = [code2d]

            codevec = get_x_vec(code3d, word2vec)
            X_test = list(map(lambda v: v[0:150], codevec[0]))

            y_pred = clf.predict(X_test) # true or false

            df['line-score-pred'] = y_pred.astype(int) # 1 or 0, if 1 then the label is correct (ie. l should have no defect and is labeled as no defect), else 0

            all_df_list.append(df)

        all_df = pd.concat(all_df_list)

        all_df.to_csv(result_dir+rel+'-line-lvl-result.csv',index=False)

        print('finished',rel)



def hyperparameters_tuning(dataset_name, Classifier, params_grid):
    word2vec = get_W2V(dataset_name)
    X_train, y_train = get_XY_train(dataset_name, word2vec)    
    clf_search = Classifier()
    search_cv = RandomizedSearchCV(
        estimator=clf_search,
        param_distributions=params_grid,
        scoring='f1',
        n_iter=25,
        n_jobs=4,
        verbose=1
    )
    
    search_cv.fit(X_train, y_train)
    return search_cv.best_params_

def dict_to_csv(dict, path):
    pd.DataFrame.from_dict(dict).to_csv(path, index=False)

rf_params_grid = {
    "n_estimators": np.array([10, 50, 200, 1000]),
    "max_depth": np.array([3, 7, 12, 15]),
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
    "max_features": np.array([1, "sqrt"]),
    "max_samples": np.array([10000])
}

# https://xgboost.readthedocs.io/en/stable/parameter.html
xgb_params_grid = {    
    "objective": ["binary:logistic", "reg:logistic", "count:poisson"],
    "n_estimators": np.array([10, 50, 200, 1000]),
    "tree_method": ["hist"],
    "max_depth": np.array([3, 7, 12, 15]),
    "learning_rate": np.arange(0.2, 0.9, 0.1),
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'colsample_bylevel': np.arange(0.5, 1.0, 0.1),
}

# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
lgbm_params_grid = {
    "objective": np.array(["regression", "binary", "gamma"]),
    "n_estimators": np.array([10, 50, 200, 1000]),
    "learning_rate": np.arange(0.2, 0.9, 0.1),
    "num_leaves": np.arange(20, 3000, 20),
    "max_depth": np.array([3, 7, 12, 15]),
    "min_data_in_leaf": np.arange(200, 10000, 100),
    "lambda_l1": np.arange(0, 100, 5),
    "lambda_l2":  np.arange(0, 100, 5),
    "min_gain_to_split": np.array([0, 5, 10]),
    "bagging_fraction": np.arange(0.2, 0.95, 0.1),
    "bagging_freq": np.array([1]),
    "feature_fraction": np.arange(0.2, 0.95, 0.1)
}

classifier_by_name = {
    "RF": { "classifier": RandomForestClassifier, "params_grid": rf_params_grid },
    "XGB": { "classifier": XGBClassifier, "params_grid": xgb_params_grid },
    "LGBM": { "classifier": LGBMClassifier, "params_grid": lgbm_params_grid }
}

default_proj_name = "activemq"
proj_name = sys.argv[1] if sys.argv.length > 0 else default_proj_name
results = { "RF": {}, "XGB": {}, "LGBM": {} }
results_path = "../../output/decision-tree-best-params.csv"

for classifier_name in classifier_by_name.keys():
    Classifier = classifier_by_name[classifier_name]["classifier"]
    best_params = hyperparameters_tuning(proj_name, Classifier, classifier_by_name[classifier_name]["params_grid"])
    print(best_params, "best params")
    results[classifier_name]["best_params"] = best_params
    train_and_predict(proj_name, classifier_name, Classifier, best_params)

dict_to_csv(results, results_path)
print(results, "Best params")
