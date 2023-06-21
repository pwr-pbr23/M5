import pandas as pd
from math import sqrt
import os, argparse

arg = argparse.ArgumentParser()

# arg.add_argument('-dataset',type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-classifiers',type=str, default='all', help='list of calssifiers. Values = all,dp,rf,xgb,lgbm,bi_lstm,bow,cnn,dbn')

args = arg.parse_args()

classifiers = args.classifiers

deep_line_dp_prediction_folder = '../output/prediction/DeepLineDP/within-release/'
deep_line_dp_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv'] # only activemq
# deepLineDp_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv', 'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

rf_prediction_folder = '../output/RF-line-level-result/'
rf_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv'] # only activemq
# rf_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv', 'camel-2.10.0-line-lvl-result.csv', 'camel-2.11.0-line-lvl-result.csv', 'derby-10.5.1.1-line-lvl-result.csv', 'groovy-1_6_BETA_2-line-lvl-result.csv', 'hbase-0.95.2-line-lvl-result.csv', 'hive-0.12.0-line-lvl-result.csv', 'jruby-1.5.0-line-lvl-result.csv', 'jruby-1.7.0.preview1-line-lvl-result.csv', 'lucene-3.0.0-line-lvl-result.csv', 'lucene-3.1-line-lvl-result.csv', 'wicket-1.5.3-line-lvl-result.csv']

xgb_prediction_folder = '../output/XGB-line-level-result/'
xgb_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv'] # only activemq
# xgb_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv', 'camel-2.10.0-line-lvl-result.csv', 'camel-2.11.0-line-lvl-result.csv', 'derby-10.5.1.1-line-lvl-result.csv', 'groovy-1_6_BETA_2-line-lvl-result.csv', 'hbase-0.95.2-line-lvl-result.csv', 'hive-0.12.0-line-lvl-result.csv', 'jruby-1.5.0-line-lvl-result.csv', 'jruby-1.7.0.preview1-line-lvl-result.csv', 'lucene-3.0.0-line-lvl-result.csv', 'lucene-3.1-line-lvl-result.csv', 'wicket-1.5.3-line-lvl-result.csv']

lgbm_prediction_folder = '../output/LGBM-line-level-result/'
lgbm_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv'] # only activemq
# lgbm_prediction_files = ['activemq-5.2.0-line-lvl-result.csv', 'activemq-5.3.0-line-lvl-result.csv', 'activemq-5.8.0-line-lvl-result.csv', 'camel-2.10.0-line-lvl-result.csv', 'camel-2.11.0-line-lvl-result.csv', 'derby-10.5.1.1-line-lvl-result.csv', 'groovy-1_6_BETA_2-line-lvl-result.csv', 'hbase-0.95.2-line-lvl-result.csv', 'hive-0.12.0-line-lvl-result.csv', 'jruby-1.5.0-line-lvl-result.csv', 'jruby-1.7.0.preview1-line-lvl-result.csv', 'lucene-3.0.0-line-lvl-result.csv', 'lucene-3.1-line-lvl-result.csv', 'wicket-1.5.3-line-lvl-result.csv']

bi_lstm_prediction_folder = '../output/prediction/Bi-LSTM/within-release/'
bi_lstm_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv'] # only activemq
# bi_lstm_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv', 'camel-2.10.0-6-epochs.csv', 'camel-2.11.0-6-epochs.csv', 'derby-10.5.1.1-6-epochs.csv', 'groovy-1_6_BETA_2-6-epochs.csv', 'hbase-0.95.2-6-epochs.csv', 'hive-0.12.0-6-epochs.csv', 'jruby-1.5.0-6-epochs.csv', 'jruby-1.7.0.preview1-6-epochs.csv', 'lucene-3.0.0-6-epochs.csv', 'lucene-3.1-6-epochs.csv', 'wicket-1.5.3-6-epochs.csv']

bow_prediction_folder = '../output/prediction/BoW/within-release/'
bow_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv'] # only activemq
# bow_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv', 'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

cnn_prediction_folder = '../output/prediction/CNN/within-release/'
cnn_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv'] # only activemq
# cnn_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv', 'camel-2.10.0-6-epochs.csv', 'camel-2.11.0-6-epochs.csv', 'derby-10.5.1.1-6-epochs.csv', 'groovy-1_6_BETA_2-6-epochs.csv', 'hbase-0.95.2-6-epochs.csv', 'hive-0.12.0-6-epochs.csv', 'jruby-1.5.0-6-epochs.csv', 'jruby-1.7.0.preview1-6-epochs.csv', 'lucene-3.0.0-6-epochs.csv', 'lucene-3.1-6-epochs.csv', 'wicket-1.5.3-6-epochs.csv']

dbn_prediction_folder = '../output/prediction/DBN/within-release/'
dbn_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv'] # only activemq
# dbn_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv', 'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

thresholds = [0.99, 0.9, 0.8, 0.6, 0.3, 0.1, 0.01, 0.0001, 0]

def get_eval_data_frames(prediction_folder, prediction_files):
    df_all = pd.DataFrame()
    for file_name in prediction_files:
        new_df = pd.read_csv(prediction_folder + file_name)
        df_all = pd.concat([df_all, new_df])
    return df_all

def get_confusion_matrix_deepLine(df, threshold):
    # True Positive (TP)
    tp = len(df[(df['line-level-ground-truth'] == True) & (df['line-attention-score'] > threshold)])
    
    # True Negative (TN)
    tn = len(df[(df['line-level-ground-truth'] == False) & (df['line-attention-score'] <= threshold)])

    # False Positive (FP)
    fp = len(df[(df['line-level-ground-truth'] == False) & (df['line-attention-score'] > threshold)])

    # False Negative (FN)
    fn = len(df[(df['line-level-ground-truth'] == True) & (df['line-attention-score'] <= threshold)])

    return tp, tn, fp, fn

def get_confusion_matrix_tree_clasifier(df):
    # True Positive (TP)
    tp_df = df[(df['line-label'] == True) & (df['line-score-pred'] == 1)]
    tp = 0 if tp_df.empty else len(tp_df)

    # True Negative (TN)
    tn_df = df[(df['line-label'] == False) & (df['line-score-pred'] == 0)]
    tn = 0 if tn_df.empty else len(tn_df)

    # False Positive (FP)
    fp_df = df[(df['line-label'] == False) & (df['line-score-pred'] == 1)]
    fp = 0 if fp_df.empty else len(fp_df)

    # False Negative (FN)
    fn_df = df[(df['line-label'] == True) & (df['line-score-pred'] == 0)]
    fn = 0 if fn_df.empty else len(fn_df)

    return tp, tn, fp, fn

def get_confusion_matrix_file_level_baseline(df):
    # True Positive (TP)
    tp_df = df[(df['file-level-ground-truth'] == True) & (df['prediction-label'] == True)]
    tp = 0 if tp_df.empty else len(tp_df)

    # True Negative (TN)
    tn_df = df[(df['file-level-ground-truth'] == False) & (df['prediction-label'] == False)]
    tn = 0 if tn_df.empty else len(tn_df)

    # False Positive (FP)
    fp_df = df[(df['file-level-ground-truth'] == False) & (df['prediction-label'] == True)]
    fp = 0 if fp_df.empty else len(fp_df)

    # False Negative (FN)
    fn_df = df[(df['file-level-ground-truth'] == True) & (df['prediction-label'] == False)]
    fn = 0 if fn_df.empty else len(fn_df)

    return tp, tn, fp, fn

def calculate_mcc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator != 0 else 0

    return mcc

def calculate_balanced_accuracy(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy

def evaluate_metrics(confusion_matrix_resolver, df, baseline_name):
    tp, tn, fp, fn = confusion_matrix_resolver(df)
    mcc = calculate_mcc(tp, tn, fp, fn)
    ba = calculate_balanced_accuracy(tp, tn, fp, fn)

    print()
    print(baseline_name, ":")
    print("MCC = ", mcc)
    print("BA = ", ba)

def evaluate_metrics_for_tree_clasifiers(prefiction_file, baseline_name):
    evaluate_metrics(get_confusion_matrix_tree_clasifier, prefiction_file, baseline_name)

def evaluate_metrics_for_baselines(prefiction_file, baseline_name):
    evaluate_metrics(get_confusion_matrix_file_level_baseline, prefiction_file, baseline_name)

print("MCC: Matthews Correlation Coefficient")
print("RF: Balanced Accuracy")

def try_process_clasifier(key, name, metrix_resolver, prediction_folder, prediction_files):
    if "all" in classifiers or key in classifiers:
        df_rf = get_eval_data_frames(prediction_folder, prediction_files)
        metrix_resolver(df_rf, name)
    else:
        print(name, "not included in statistics")

try_process_clasifier("rf", "Random Forest (line level)", evaluate_metrics_for_tree_clasifiers, rf_prediction_folder, rf_prediction_files)
try_process_clasifier("xgb", "XGBoost (line level)", evaluate_metrics_for_tree_clasifiers, xgb_prediction_folder, xgb_prediction_files)
try_process_clasifier("lgbm", "LightGBM (line level)", evaluate_metrics_for_tree_clasifiers, lgbm_prediction_folder, lgbm_prediction_files)
try_process_clasifier("bi_lstm", "Bi-LSTM (file level)", evaluate_metrics_for_baselines, bi_lstm_prediction_folder, bi_lstm_prediction_files)
try_process_clasifier("bow", "BOW (file level)", evaluate_metrics_for_baselines, bow_prediction_folder, bow_prediction_files)
try_process_clasifier("cnn", "CNN (file level)", evaluate_metrics_for_baselines, cnn_prediction_folder, cnn_prediction_files)
try_process_clasifier("dbn", "DBN (file level)", evaluate_metrics_for_baselines, dbn_prediction_folder, dbn_prediction_files)

def get_dp_mcc_with_thresholds(df, thresholds):
    for t in thresholds:
        resolver = lambda df: get_confusion_matrix_deepLine(df, t)
        evaluate_metrics(resolver, df, "DeepLineDP (line level) with Threshold " + str(t))

df_dp = get_eval_data_frames(deep_line_dp_prediction_folder, deep_line_dp_prediction_files)
get_dp_mcc_with_thresholds(df_dp, thresholds)