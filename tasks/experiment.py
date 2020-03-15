import argparse
import itertools
import json
import logging
import pprint
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Dict
from typing import List

import mlflow
import numpy as np
import pandas as pd
from const import SUBJECTS, LABELS_2
from joblib import Parallel, delayed
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from scipy.stats import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from eeglibrary.src.eeg import EEG


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--n-test-users', default=2, type=int)

    return parser


def label_func(row):
    return LABELS_2[row[0].split('_')[-1][:-4]]


def load_func(row):
    return EEG.load_pkl(row[0]).values


def set_inner_subj_data_paths(expt_dir, expt_conf, test_user_ids) -> Dict:
    manifest_df = pd.read_csv(expt_conf['manifest_path'], header=None)



    # This split rate has no effect if you specify group k-fold. Train and val set will be combined on CV
    train_data_df.iloc[:int(len(train_data_df) // 2)].to_csv(expt_dir / 'train_data.csv', index=False, header=None)
    train_data_df.iloc[int(len(train_data_df) // 2):].to_csv(expt_dir / 'val_data.csv', index=False, header=None)
    expt_conf['train_path'] = str(expt_dir / 'train_data.csv')
    expt_conf['val_path'] = str(expt_dir / 'val_data.csv')

    test_df = manifest_df[user_extracted.isin(test_user_ids)]
    test_df.to_csv(expt_dir / 'test_data.csv', index=False, header=None)
    expt_conf['test_path'] = str(expt_dir / 'test_data.csv')

    return expt_conf


def set_inter_subj_data_paths(expt_dir, expt_conf, test_user_ids) -> Dict:
    manifest_df = pd.read_csv(expt_conf['manifest_path'], header=None)
    user_extracted = manifest_df.apply(lambda x: x[0].split('/')[-4], axis=1)
    train_data_df = manifest_df[~user_extracted.isin(test_user_ids)]
    # This split rate has no effect if you specify group k-fold. Train and val set will be combined on CV
    train_data_df.iloc[:int(len(train_data_df) // 2)].to_csv(expt_dir / 'train_data.csv', index=False, header=None)
    train_data_df.iloc[int(len(train_data_df) // 2):].to_csv(expt_dir / 'val_data.csv', index=False, header=None)
    expt_conf['train_path'] = str(expt_dir / 'train_data.csv')
    expt_conf['val_path'] = str(expt_dir / 'val_data.csv')

    test_df = manifest_df[user_extracted.isin(test_user_ids)]
    test_df.to_csv(expt_dir / 'test_data.csv', index=False, header=None)
    expt_conf['test_path'] = str(expt_dir / 'test_data.csv')

    return expt_conf


def get_inter_cv_groups(expt_conf, test_user_ids: List[int]):
    manifest_df = pd.read_csv(expt_conf['manifest_path'], header=None)
    user_extracted = manifest_df.apply(lambda x: x[0].split('/')[-4], axis=1)
    user_extracted = user_extracted[~user_extracted.isin(test_user_ids)]

    subjects = pd.DataFrame(user_extracted[~user_extracted.isin(test_user_ids)].unique())
    subjects['group'] = [j % expt_conf['n_splits'] for j in range(len(subjects))]
    subjects = subjects.set_index(0)
    groups = user_extracted.apply(lambda x: subjects.loc[x, 'group'])
    return groups


def dump_dict(path, dict_):
    with open(path, 'w') as f:
        json.dump(dict_, f, indent=4)


def main(expt_conf, expt_dir, hyperparameters, typical_train_func, test_user_ids):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = [0, 1]
    if expt_conf['train_manager'] == 'nn':
        metrics_names = {'train': ['loss', 'uar'],
                         'val': ['loss', 'uar'],
                         'test': ['loss', 'uar']}
    else:
        metrics_names = {'train': ['uar'],
                         'val': ['uar'],
                         'test': ['uar']}

    dataset_cls = ManifestWaveDataSet
    if expt_conf['inter_subj']:
        expt_conf = set_inter_subj_data_paths(expt_dir, expt_conf, test_user_ids)
    else:
        expt_conf = set_inner_subj_data_paths(expt_dir, expt_conf, test_user_ids)
    process_func = None

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    if expt_conf['inter_subj']:
        groups = get_inter_cv_groups(expt_conf, test_user_ids)
    else:
        groups = get_inner_cv_groups(expt_conf, test_user_ids)

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        # TODO cv時はmean と stdをtrainとvalの分割後に求める必要がある
        with mlflow.start_run():
            result_series, val_pred = typical_train_func(expt_conf, load_func, label_func, process_func, dataset_cls, groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_parallel'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = patterns
    result_list = np.array([result for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, _), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    # Train with train + devel dataset
    if expt_conf['test']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        metrics, pred_dict_list = typical_experiment(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                     groups)

        if expt_conf['return_prob']:
            ensemble_pred = np.argmax(np.array([pred_dict['test'] for pred_dict in pred_dict_list]).sum(axis=0), axis=1)
        else:
            ensemble_pred = stats.mode(np.array([pred_dict['test'] for pred_dict in pred_dict_list]), axis=0)[0][0]
        test_labels = pd.read_csv(expt_conf['test_path'], header=None).apply(label_func, axis=1)
        uar = balanced_accuracy_score(test_labels, ensemble_pred)
        print(f'{uar:.05f}')
        print(f'Confusion matrix: \n{confusion_matrix(test_labels, ensemble_pred)}')
        test_pred_name = f"sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}_{uar:.04f}.csv"
        pd.DataFrame(ensemble_pred).to_csv(expt_dir / test_pred_name, index=False)
        print(f"Test prediction file is saved in {expt_dir / test_pred_name}")

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.DEBUG)
    logging.getLogger("ml").addHandler(console)

    hyperparameters = {
        'lr': [0.01],
    }

    # manifest_df = pd.read_csv(expt_conf['manifest_path'], header=None)
    # manifest_df = manifest_df[manifest_df.apply(lambda x: x[0].split('_')[-1][:-4], axis=1) != 'ictal']
    # user_extracted = manifest_df.apply(lambda x: x[0].split('/')[-4], axis=1)
    # manifest_df = manifest_df[user_extracted.isin(SUBJECTS)]
    # manifest_df.to_csv('/media/tomoya/SSD-PGU3/research/brain/chb-mit/manifest.csv', header=None, index=False)
    # exit()

    expt_conf['inter_subj'] = 'interictal_preictal' not in expt_conf['manifest_path'].split('/')

    test_user_id_patterns = np.array(SUBJECTS[:2]).reshape((-1, expt_conf['n_test_users']))
    test_folder_name = f"ntest-{expt_conf['n_test_users']}"

    for test_user_id_pattern in test_user_id_patterns:
        expt_conf['expt_id'] = f"{expt_conf['model_type']}"
        pj_dir = Path(__file__).resolve().parents[1]
        expt_dir = pj_dir / 'output' / test_folder_name / '_'.join(test_user_id_pattern) / f"{expt_conf['expt_id']}"
        expt_dir.mkdir(exist_ok=True, parents=True)
        main(expt_conf, expt_dir, hyperparameters, typical_train, test_user_id_pattern)
        # break

    # aggregate(expt_conf)

    if expt_conf['expt_id'] == 'debug':
        import shutil

        shutil.rmtree('../mlruns')
