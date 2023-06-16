import pandas as pd
from math import sqrt

deepLineDp_prediction_file = '../output/prediction/DeepLineDP/within-release/activemq-5.2.0.csv'
rf_prediction_file = '../output/RF-line-level-result/activemq-5.2.0-line-lvl-result.csv'
bow_prediction_file = '../output/prediction/BoW/within-release/activemq-5.2.0.csv'

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

def calculate_confusion_matrix_rf(df):
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

def calculate_confusion_matrix_bow(df):
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

df = pd.read_csv(rf_prediction_file)
tp, tn, fp, fn = calculate_confusion_matrix_rf(df)
mcc = calculate_mcc(tp, tn, fp, fn)
ba = calculate_balanced_accuracy(tp, tn, fp, fn)

print("Matthews Correlation Coefficient (MCC) RF:", mcc)
print("Balanced Accuracy (BA) RF:", ba)

df = pd.read_csv(bow_prediction_file)
tp, tn, fp, fn = calculate_confusion_matrix_bow(df)
mcc = calculate_mcc(tp, tn, fp, fn)
ba = calculate_balanced_accuracy(tp, tn, fp, fn)

print("Matthews Correlation Coefficient (MCC) BoW:", mcc)
print("Balanced Accuracy (BA) BoW:", ba)

def get_dp_mcc_with_thresholds(df, thresholds):
    for t in thresholds:
        tp, tn, fp, fn = calculate_confusion_matrix_deepLine(df, t)
        mcc = calculate_mcc(tp, tn, fp, fn)
        ba = calculate_balanced_accuracy(tp, tn, fp, fn)
        print("Matthews Correlation Coefficient (MCC) DP:", mcc, "Threshold", t)
        print("Balanced Accuracy (BA) DP:", ba, "Threshold", t)


df = pd.read_csv(deepLineDp_prediction_file)
get_dp_mcc_with_thresholds(df, thresholds)
