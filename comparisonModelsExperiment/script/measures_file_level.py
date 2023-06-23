import pandas as pd
from math import sqrt
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

arg = argparse.ArgumentParser()

arg.add_argument('-classifiers', type=str, default='bi_lstm,bow,cnn,dbn',
                 help='list of calssifiers. Values = all,bi_lstm,bow,cnn,dbn')
arg.add_argument('-dataset', type=str, default='all',
                 help='project name. Values = activemq')

args = arg.parse_args()

classifiers = args.classifiers
dataset = args.dataset

print("Dataset: ", dataset)

deep_line_dp_prediction_folder = '../output/prediction/DeepLineDP/within-release/'
if dataset == 'activemq':
    deep_line_dp_prediction_files = [
        'activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv']  # only activemq
else:
    deep_line_dp_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv',
                                     'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

bi_lstm_prediction_folder = '../output/prediction/Bi-LSTM/within-release/'
if dataset == 'activemq':
    bi_lstm_prediction_files = ['activemq-5.2.0-6-epochs.csv',
                                'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv']  # only activemq
else:
    bi_lstm_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv', 'camel-2.10.0-6-epochs.csv', 'camel-2.11.0-6-epochs.csv', 'derby-10.5.1.1-6-epochs.csv',
                                'groovy-1_6_BETA_2-6-epochs.csv', 'hbase-0.95.2-6-epochs.csv', 'hive-0.12.0-6-epochs.csv', 'jruby-1.5.0-6-epochs.csv', 'jruby-1.7.0.preview1-6-epochs.csv', 'lucene-3.0.0-6-epochs.csv', 'lucene-3.1-6-epochs.csv', 'wicket-1.5.3-6-epochs.csv']

bow_prediction_folder = '../output/prediction/BoW/within-release/'
if dataset == 'activemq':
    bow_prediction_files = [
        'activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv']  # only activemq
else:
    bow_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv',
                            'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

cnn_prediction_folder = '../output/prediction/CNN/within-release/'
if dataset == 'activemq':
    cnn_prediction_files = ['activemq-5.2.0-6-epochs.csv',
                            'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv']  # only activemq
else:
    cnn_prediction_files = ['activemq-5.2.0-6-epochs.csv', 'activemq-5.3.0-6-epochs.csv', 'activemq-5.8.0-6-epochs.csv', 'camel-2.10.0-6-epochs.csv', 'camel-2.11.0-6-epochs.csv', 'derby-10.5.1.1-6-epochs.csv', 'groovy-1_6_BETA_2-6-epochs.csv',
                            'hbase-0.95.2-6-epochs.csv', 'hive-0.12.0-6-epochs.csv', 'jruby-1.5.0-6-epochs.csv', 'jruby-1.7.0.preview1-6-epochs.csv', 'lucene-3.0.0-6-epochs.csv', 'lucene-3.1-6-epochs.csv', 'wicket-1.5.3-6-epochs.csv']

dbn_prediction_folder = '../output/prediction/DBN/within-release/'
if dataset == 'activemq':
    dbn_prediction_files = [
        'activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv']  # only activemq
else:
    dbn_prediction_files = ['activemq-5.2.0.csv', 'activemq-5.3.0.csv', 'activemq-5.8.0.csv', 'camel-2.10.0.csv', 'camel-2.11.0.csv', 'derby-10.5.1.1.csv',
                            'groovy-1_6_BETA_2.csv', 'hbase-0.95.2.csv', 'hive-0.12.0.csv', 'jruby-1.5.0.csv', 'jruby-1.7.0.preview1.csv', 'lucene-3.0.0.csv', 'lucene-3.1.csv', 'wicket-1.5.3.csv']

thresholds = [0.99, 0.9, 0.8, 0.6, 0.3, 0.1, 0.01, 0.0001, 0]


def get_eval_data_frames(prediction_folder, prediction_files, is_line_level=False):
    df_all = pd.DataFrame()
    for file_name in prediction_files:
        new_df = pd.read_csv(prediction_folder + file_name)

        if is_line_level:
            new_df = new_df.groupby("filename").agg({"file-level-ground-truth": "first", "line-attention-score": "max"}).reset_index()
            new_df.rename(columns={"line-attention-score": "max-line-attention-score"}, inplace=True)

        df_all = pd.concat([df_all, new_df])
    return df_all


def get_confusion_matrix_deepLine(df, threshold):
    # True Positive (TP)
    tp = len(df[(df['file-level-ground-truth'] == True)
             & (df['max-line-attention-score'] > threshold)])

    # True Negative (TN)
    tn = len(df[(df['file-level-ground-truth'] == False)
             & (df['max-line-attention-score'] <= threshold)])

    # False Positive (FP)
    fp = len(df[(df['file-level-ground-truth'] == False)
             & (df['max-line-attention-score'] > threshold)])

    # False Negative (FN)
    fn = len(df[(df['file-level-ground-truth'] == True) &
             (df['max-line-attention-score'] <= threshold)])

    return tp, tn, fp, fn


def get_confusion_matrix_file_level_baseline(df):
    # True Positive (TP)
    tp_df = df[(df['file-level-ground-truth'] == True)
               & (df['prediction-label'] == True)]
    tp = 0 if tp_df.empty else len(tp_df)

    # True Negative (TN)
    tn_df = df[(df['file-level-ground-truth'] == False)
               & (df['prediction-label'] == False)]
    tn = 0 if tn_df.empty else len(tn_df)

    # False Positive (FP)
    fp_df = df[(df['file-level-ground-truth'] == False)
               & (df['prediction-label'] == True)]
    fp = 0 if fp_df.empty else len(fp_df)

    # False Negative (FN)
    fn_df = df[(df['file-level-ground-truth'] == True)
               & (df['prediction-label'] == False)]
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

    return mcc, ba


def generate_charts(classifiers, metric, values):
    fig, ax = plt.subplots()
    index = range(len(classifiers))

    ax.bar(index, values)
    ax.set_xlabel('Classifiers')
    ax.set_ylabel(metric)
    ax.set_title(f'Comparison of {metric} between Classifiers')
    ax.set_xticks(index)
    ax.set_xticklabels(classifiers, rotation=45)

    # Find the classifier with the highest MCC
    best_classifier = classifiers[values.index(max(values))]
    ax.axhline(max(values), color='r', linestyle='--',
               label=f'Best Classifier: {best_classifier}')
    ax.legend()

    plt.tight_layout()
    
    with PdfPages(f'../output/figure/file-level-{metric}.pdf') as pdf:
        pdf.savefig(fig)

    plt.close(fig) 


def evaluate_metrics_for_baselines(prefiction_file, baseline_name):
    return evaluate_metrics(get_confusion_matrix_file_level_baseline,
                            prefiction_file, baseline_name)


def try_process_clasifier(key, name, metrix_resolver, prediction_folder, prediction_files):
    if "all" in classifiers or key in classifiers:
        df_classif = get_eval_data_frames(prediction_folder, prediction_files)

        return metrix_resolver(df_classif, name)
    else:
        print(name, "not included in statistics")

def get_dp_results_with_thresholds(df, thresholds, results):
    for t in thresholds:
        def resolver(df): return get_confusion_matrix_deepLine(df, t)
        mcc, ba = evaluate_metrics(
            resolver, df, "DeepLineDP (line level) with Threshold " + str(t))
        results["dpline_" + str(t)] = {}
        results["dpline_" +
                str(t)]['MCC'] = mcc
        results["dpline_" +
                str(t)]['Balanced Accuracy'] = ba


def get_classifiers_results(classifiers, results):
    for classifier in classifiers:
        results[classifier] = {}
        mcc, ba = 0, 0
        if classifier == 'bi_lstm':
            mcc, ba = try_process_clasifier(
                "bi_lstm", "Bi-LSTM (file level)", evaluate_metrics_for_baselines, bi_lstm_prediction_folder, bi_lstm_prediction_files)
        elif classifier == 'bow':
            mcc, ba = try_process_clasifier(
                "bow", "BOW (file level)", evaluate_metrics_for_baselines, bow_prediction_folder, bow_prediction_files)
        elif classifier == 'cnn':
            mcc, ba = try_process_clasifier(
                "cnn", "CNN (file level)", evaluate_metrics_for_baselines, cnn_prediction_folder, cnn_prediction_files)
        elif classifier == 'dbn':
            mcc, ba = try_process_clasifier(
                "dbn", "DBN (file level)", evaluate_metrics_for_baselines, dbn_prediction_folder, dbn_prediction_files)

        results[classifier]['MCC'] = mcc
        results[classifier]['Balanced Accuracy'] = ba


print("MCC: Matthews Correlation Coefficient")
print("BA: Balanced Accuracy")
results = {}
classifiersList = classifiers.split(',')
df_dp = get_eval_data_frames(
    deep_line_dp_prediction_folder, deep_line_dp_prediction_files, is_line_level=True)

get_classifiers_results(classifiersList, results)
get_dp_results_with_thresholds(df_dp, thresholds, results)

# Generate charts
metrics = ['MCC', 'Balanced Accuracy']
classifiersResultList = list(results.keys())
values = {metric: [results[classifier][metric]
                   for classifier in classifiersResultList] for metric in metrics}

# Plotting
for metric in metrics:
    generate_charts(classifiersResultList, metric, values[metric])
