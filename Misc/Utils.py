import os
import pandas as pd
from typing import List, Any
from Misc.ParsedData import ParsedData
from Misc.ModelInput import ModelInput
import glob
from Chunkers import ChunkByVolume, ChunkByCommands
from Preprocessing import CLTTokenizer, PLTTokenizer, RFFeaturizer, DeftPunkFeaturizer
from Engines.TransformerManager import TransformerManager
from Engines.TreeManager import TreeManager
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit
import scipy.stats as stats
import math


def read_data(data_path, model_name) -> List[ParsedData]:
    metadata_path = os.path.join(data_path, 'metadata.csv')
    df_metadata = pd.read_csv(metadata_path)
    df_metadata.set_index('Recording Index', inplace=True)
    parsed_data_list = []

    #  RW
    rw_path = os.path.join(data_path, 'Ransomware')
    rw_parquet_files = glob.glob(f"{rw_path}/*.parquet")

    for rw_file in rw_parquet_files:
        rec_name = os.path.splitext(os.path.basename(rw_file))[0]
        current_data = pd.read_parquet(rw_file, engine='pyarrow')
        adapted_current_data = adapt_data(current_data)
        train_test = "Train" if df_metadata.loc[rec_name, 'Train/Test'] == 'Train' else "Test"
        if model_name == 'CLT':
            rec_max_slba = adapted_current_data.SLBA.max()
            current_parsed_data = ParsedData(1, adapted_current_data, train_test,
                                             rec_number=int(rec_name.split('_')[1]),
                                             metadata=rec_max_slba)  # Per recording label
        else:
            current_parsed_data = ParsedData(1, adapted_current_data, train_test,
                                             rec_number=int(rec_name.split('_')[1]))  # Per recording level
        parsed_data_list.append(current_parsed_data)

    #  Benign
    rw_path = os.path.join(data_path, 'Benign')
    rw_parquet_files = glob.glob(f"{rw_path}/*.parquet")

    for rw_file in rw_parquet_files:
        rec_name = os.path.splitext(os.path.basename(rw_file))[0]
        current_data = pd.read_parquet(rw_file, engine='pyarrow')
        adapted_current_data = adapt_data(current_data)
        train_test = "Train" if df_metadata.loc[rec_name, 'Train/Test'] == 'Train' else "Test"
        if model_name == 'CLT':
            rec_max_slba = adapted_current_data.SLBA.max()
            current_parsed_data = ParsedData(0, adapted_current_data, train_test,
                                             rec_number=int(rec_name.split('_')[1]),
                                             metadata=rec_max_slba)  # Per recording label
        else:
            current_parsed_data = ParsedData(0, adapted_current_data, train_test,
                                             rec_number=int(rec_name.split('_')[1]))  # Per recording level

        parsed_data_list.append(current_parsed_data)

    return parsed_data_list


def adapt_data(df_recording):
    df_recording_adapted = df_recording.rename(columns={"Offset": "SLBA", "Size": "NLB"})

    df_recording_adapted["SLBA"] //= 512
    df_recording_adapted["NLB"] //= 512
    df_recording_adapted["WaR"] //= 512
    df_recording_adapted["RaR"] //= 512
    df_recording_adapted["RaW"] //= 512
    df_recording_adapted["WaW"] //= 512

    return df_recording_adapted


def run_chunker(parsed_data_list: List[ParsedData], model: str):
    data_with_chunks_list = []

    if model == 'CLT':
        chunker = ChunkByCommands.ChunkByCommands(parsed_data_list)

    else:
        chunker = ChunkByVolume.ChunkByVolume(parsed_data_list)

    for parsed_data in parsed_data_list:
        parsed_df = parsed_data.data

        chunks_indices: pd.Series = chunker.create_chunk_indices_series(parsed_df)

        # Check if the indices of the Series and DataFrame are the same
        if not parsed_df.index.equals(chunks_indices.index):
            raise ValueError("Index mismatch: DataFrame and Series indices do not align")

        parsed_df['Chunk_Index'] = chunks_indices

        #  Group by chunk index
        grouped_parsed_df = parsed_df.groupby('Chunk_Index')

        for ck_idx, group in grouped_parsed_df:
            if ck_idx == -1:  # Not a chunk
                continue
            chunk_df = group.drop(columns='Chunk_Index').reset_index(drop=True)
            parsed_chunk_data = ParsedData(parsed_data.label, chunk_df,
                                           parsed_data.train_test,
                                           parsed_data.rec_number,
                                           parsed_data.metadata)  # Still, provisional per-recording label. The label itself will be added at the preprocessing stage
            data_with_chunks_list.append(parsed_chunk_data)

    return data_with_chunks_list


def run_preprocessing(data_with_chunks_list: List[ParsedData], model: str):
    model_input_list = []

    if model == 'CLT':
        preprocessor = CLTTokenizer.CLTTokenizer()
    elif model == 'PLT':
        preprocessor = PLTTokenizer.PLTTokenizer()
    elif model == 'RF':
        preprocessor = RFFeaturizer.RFFeaturizer()
    else:
        preprocessor = DeftPunkFeaturizer.DeftPunkFeaturizer()

    for chunk_data_element in data_with_chunks_list:
        current_features = preprocessor.generate_per_chunk(chunk_data_element)

        #  Update chunk label
        if model == 'PLT':
            current_chunk_label = get_chunk_label(chunk_data_element.data, model, preprocessor)
        else:
            current_chunk_label = get_chunk_label(chunk_data_element.data, model)
        chunk_data_element.label = current_chunk_label

        current_model_input = ModelInput(chunk_data_element, current_features)

        model_input_list.append(current_model_input)

    return model_input_list


def get_chunk_label(chunk_df, model, preprocessor=None):
    if model == 'RF' or model == 'DeftPunk':
        if chunk_df.Label.any():
            return 1
        return 0
    elif model == 'CLT':
        return chunk_df.Label
    else:
        return preprocessor.get_plt_labels(chunk_df)


def train_and_infer(data_with_features_list: List[ModelInput], model: str, demo_mode: bool):
    if model == 'CLT' or model == 'PLT':
        mdl_mgr = TransformerManager(model, data_with_features_list, demo_mode)
    else:
        mdl_mgr = TreeManager(model, data_with_features_list, demo_mode)

    mdl_mgr.train()

    results_split = mdl_mgr.inference()

    results = obtain_results(results_split)

    return results


def divide_train_test(dataset: List[ModelInput]):
    train_set = []
    test_set = []

    for element in dataset:
        if element.parsed_data_element.train_test == "Train":
            train_set.append(element)
        else:
            test_set.append(element)

    return train_set, test_set


def obtain_features_and_labels(dataset):
    if type(dataset) != list:
        X = np.array(dataset.features)
        y = np.array(dataset.parsed_data_element.label)
    else:
        X = np.array([element.features for element in dataset])
        y = np.array([element.parsed_data_element.label for element in dataset])

    return X, y


def fracreg_to_label(batch_labels, model_name):
    if model_name == "PLT":
        lab = batch_labels[:, 0:2, :].sum(axis=1).mean(axis=1) > 0
    else:  # CLT
        lab = (batch_labels.mean(axis=1) if len(
            batch_labels.shape) > 1 else batch_labels.mean()) > 0
    try:
        return lab[0]
    except:
        return lab


def fracreg_to_predict_prob(predict_prob, model_name):
    if model_name == "PLT":
        pred_prob = predict_prob[:, 0:2, :].sum(axis=1).mean(axis=1)
    else:  # CLT
        pred_prob = predict_prob.mean() if len(predict_prob.shape) > 1 else predict_prob
    try:
        return pred_prob[0]
    except:
        return pred_prob


def extract_train_and_test_for_demo_mode(dataset: List[ParsedData]):
    num_train_chunks_rw = 5
    num_train_chunks_benign = 5
    num_test_chunks_rw = 2
    num_test_chunks_benign = 2

    demo_list = []

    dataset_iter = iter(dataset)
    while num_train_chunks_rw + num_train_chunks_benign + num_test_chunks_rw + num_test_chunks_benign > 0:
        current_dataset = next(dataset_iter)
        current_train_test = current_dataset.train_test
        current_label = current_dataset.label
        if current_train_test == 'Train':
            if current_label == 0:
                if num_train_chunks_benign > 0:
                    demo_list.append(current_dataset)
                    num_train_chunks_benign -= 1
            else:
                if num_train_chunks_rw > 0:
                    demo_list.append(current_dataset)
                    num_train_chunks_rw -= 1
        else:
            if current_label == 0:
                if num_test_chunks_benign > 0:
                    demo_list.append(current_dataset)
                    num_test_chunks_benign -= 1
            else:
                if num_test_chunks_rw > 0:
                    demo_list.append(current_dataset)
                    num_test_chunks_rw -= 1

    return demo_list


def clopper_pearson(x, n, alpha=0.05):
    """
    Estimate the confidence interval for a sampled Bernoulli random variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


def benchmark_results(results_input, num_splits=1, test_size=0.67):
    gss = GroupShuffleSplit(n_splits=num_splits, test_size=test_size, random_state=0)
    results_dict = []

    results = results_input.copy()

    for i, (val_index, test_index) in enumerate(gss.split(X=None, y=None, groups=results_input['rec_idx'])):
        # Find work point based on chunk level ROC - on the validation data
        current_res_val = results.iloc[val_index, :]
        fpr, tpr, thresholds = metrics.roc_curve(current_res_val['target'], current_res_val['pred_prob'])
        precision, recall, _ = metrics.precision_recall_curve(current_res_val['target'], current_res_val['pred_prob'])

        neg_ground_truth = (current_res_val['target'] == 0).sum()
        pos_ground_truth = (current_res_val['target'] == 1).sum()
        fp = fpr * neg_ground_truth
        tp = tpr * pos_ground_truth
        fpr_upper = np.array([clopper_pearson(x, neg_ground_truth)[1] for x in fp])
        tpr_lower = np.array([clopper_pearson(x, pos_ground_truth)[0] for x in tp])

        try:
            ix = np.argmax(tpr[fpr_upper <= 0.01])
            rw_predict_th = thresholds[ix]
        except:  # Demo mode
            rw_predict_th = 0.5

        # Use threshold found on validation to generate predictions on all data
        results['pred'] = results['pred_prob'] > rw_predict_th  # 0 is benign, 1 is RW.

        current_res_test = results.iloc[test_index, :]

        df_test_results = classification_workpoint(current_res_test)

        results_dict.append({'Performance': df_test_results})

    return results_dict


def classification_workpoint(results_df):
    cm = metrics.confusion_matrix(results_df['target'],
                                  results_df['pred'],
                                  labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    neg_ground_truth = tn + fp
    pos_ground_truth = tp + fn

    fpr = fp / (neg_ground_truth + 1e-10)
    fnr = fn / (pos_ground_truth + 1e-10)

    precision = tp / (tp + fp)
    recall = 1 - fnr

    F1 = 2 * precision * recall / (precision + recall)

    fpr_lower, fpr_upper = clopper_pearson(fp, neg_ground_truth)
    fnr_lower, fnr_upper = clopper_pearson(fn, pos_ground_truth)
    precision_lower, precision_upper = clopper_pearson(tp, tp + fp)

    metrics_table = {'fpr': [fpr],
                     'fnr': [fnr],
                     'precision': [precision],
                     'recall': [recall],
                     'F1': [F1],
                     'fpr_CI_up': [fpr_upper],
                     'fpr_CI_down': [fpr_lower],
                     'fpr_CI_1s': [(fpr_upper - fpr_lower) / (2 * 1.960)],  # because 1.96 is 95% confidence interval
                     'fnr_CI_up': [fnr_upper],
                     'fnr_CI_down': [fnr_lower],
                     'fnr_CI_1s': [(fnr_upper - fnr_lower) / (2 * 1.960)],  # because 1.96 is 95% confidence interval
                     'recall_CI_1s': [(fnr_upper - fnr_lower) / (2 * 1.960)],  # fnr error is same as recall
                     'precision_CI_up': [precision_upper],
                     'precision_CI_down': [precision_lower],
                     'precision_CI_1s': [(precision_upper - precision_lower) / (2 * 1.960)]
                     # fnr error is same as recall
                     }

    return pd.DataFrame(metrics_table)


def obtain_results(results_split):
    fpr = []
    fnr = []
    recall = []
    precision = []
    f1 = []

    for res in results_split:
        current_res = res['Performance']
        fpr.append(current_res['fpr'])
        fnr.append(current_res['fnr'])
        recall.append(current_res['recall'])
        precision.append(current_res['precision'])
        f1.append(current_res['F1'])

    fpr = np.array(fpr)
    fnr = np.array(fnr)
    recall = np.array(recall)
    precision = np.array(precision)
    f1 = np.array(f1)

    results_string = f'FPR = {fpr.mean()} +/- {fpr.std()}; FNR = {fnr.mean()} +/- {fnr.mean()}; Recall = {recall.mean()} +/- {recall.mean()}; Precision = {precision.mean()} +/- {precision.mean()}; F1 = {f1.mean()} +/- {f1.mean()}'

    return results_string
