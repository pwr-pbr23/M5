import pandas as pd
from math import sqrt

deepLineDp_prediction_file = '../output/prediction/DeepLineDP/within-release/activemq-5.2.0.csv'
rf_prediction_file = '../output/RF-line-level-result/activemq-5.2.0-line-lvl-result.csv'
xgb_prediction_file = '../output/XGB-line-level-result/activemq-5.2.0-line-lvl-result.csv'
lgbm_prediction_file = '../output/LGBM-line-level-result/activemq-5.2.0-line-lvl-result.csv'
bi_lstm_prediction_file = '../output/prediction/Bi-LSTM/within-release/activemq-5.2.0-6-epochs.csv'
bow_prediction_file = '../output/prediction/BoW/within-release/activemq-5.2.0.csv'
cnn_prediction_file = '../output/prediction/CNN/within-release/activemq-5.2.0-6-epochs.csv'
dbn_prediction_file = '../output/prediction/DBN/within-release/activemq-5.2.0.csv'

thresholds = [0.99, 0.9, 0.8, 0.6, 0.3, 0.1, 0.01, 0.0001, 0]

def calculate_confusion_matrix_deepLine(df, threshold):
    # True Positive (TP)
    tp = len(df[(df['line-level-ground-truth'] == True) & (df['line-attention-score'] > threshold)])
    
    # True Negative (TN)
    tn = len(df[(df['line-level-ground-truth'] == False) & (df['line-attention-score'] <= threshold)])

    # False Positive (FP)
    fp = len(df[(df['line-level-ground-truth'] == False) & (df['line-attention-score'] > threshold)])

    # False Negative (FN)
    fn = len(df[(df['line-level-ground-truth'] == True) & (df['line-attention-score'] <= threshold)])

    return tp, tn, fp, fn

def calculate_confusion_matrix_tree_clasifier(df):
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

def calculate_confusion_matrix_file_level_baseline(df):
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

def calculate_metrics_for_tree_clasifiers(prefiction_file, baseline_name):
    df = pd.read_csv(prefiction_file)
    tp, tn, fp, fn = calculate_confusion_matrix_tree_clasifier(df)
    mcc = calculate_mcc(tp, tn, fp, fn)
    ba = calculate_balanced_accuracy(tp, tn, fp, fn)

    print()
    print(baseline_name, ":")
    print("MCC = ", mcc)
    print("BA = ", ba)

def calculate_metrics_for_baselines(prefiction_file, baseline_name):
    df = pd.read_csv(prefiction_file)
    tp, tn, fp, fn = calculate_confusion_matrix_file_level_baseline(df)
    mcc = calculate_mcc(tp, tn, fp, fn)
    ba = calculate_balanced_accuracy(tp, tn, fp, fn)

    print()
    print(baseline_name, ":")
    print("MCC = ", mcc)
    print("BA = ", ba)

print("MCC: Matthews Correlation Coefficient")
print("RF: Balanced Accuracy")

calculate_metrics_for_tree_clasifiers(rf_prediction_file, "Random Forest")
calculate_metrics_for_tree_clasifiers(xgb_prediction_file, "XGBoost")
calculate_metrics_for_tree_clasifiers(lgbm_prediction_file, "LightGBM")

calculate_metrics_for_baselines(bi_lstm_prediction_file, "Bi-LSTM")
calculate_metrics_for_baselines(bow_prediction_file, "BOW")
calculate_metrics_for_baselines(cnn_prediction_file, "CNN")
calculate_metrics_for_baselines(dbn_prediction_file, "DBN")

def get_dp_mcc_with_thresholds(df, thresholds):
    for t in thresholds:
        tp, tn, fp, fn = calculate_confusion_matrix_deepLine(df, t)
        mcc = calculate_mcc(tp, tn, fp, fn)
        ba = calculate_balanced_accuracy(tp, tn, fp, fn)

        print()
        print("DeepLineDP with Threshold", t, ":")
        print("MCC = ", mcc)
        print("BA = ", ba)


df = pd.read_csv(deepLineDp_prediction_file)
get_dp_mcc_with_thresholds(df, thresholds)
