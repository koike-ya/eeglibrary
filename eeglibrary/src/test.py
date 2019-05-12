from __future__ import print_function, division

import pandas as pd
from eeglibrary.utils import test_args
from eeglibrary.models.RNN import *
from eeglibrary.utils import init_seed, init_device, set_eeg_conf, set_dataloader, set_model
from tqdm import tqdm


def test(args, class_names):
    init_seed(args)
    device = init_device(args)
    if args.model_name in ['kneighbor', 'knn']:
        args.model_name = 'kneighbor'
    numpy = 'nn' not in args.model_name

    eeg_conf = set_eeg_conf(args)
    dataloader = set_dataloader(args, class_names, eeg_conf, phase='test', device=device)

    model = set_model(args, class_names, eeg_conf, device)
    if numpy:
        model.load_model_(args.model_path)
    else:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    pred_list = []
    path_list = []

    for i, (inputs, paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)

        if numpy:
            preds = model.predict(inputs)
        else:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        pred_list.extend(preds)
        # Transpose paths, but I don't know why dataloader outputs aukward
        path_list.extend([list(pd.DataFrame(paths).iloc[:, i].values) for i in range(len(paths[0]))])

    def ensemble_preds(pred_list, path_list, sub_df, thresh):
        # もともとのmatファイルごとに振り分け直す
        patient_name = path_list[0][0].split('/')[-3]
        orig_mat_list = sub_df[sub_df['clip'].apply(lambda x: '_'.join(x.split('_')[:2])) == patient_name]
        ensembled_pred_list = []
        for orig_mat_name in orig_mat_list['clip']:
            seg_number = int(orig_mat_name[-8:-4])
            one_segment_preds = [pred for path, pred in zip(path_list[0], pred_list) if
                                 int(path.split('/')[-2].split('_')[-1]) == seg_number]
            ensembled_pred = int(sum(one_segment_preds) >= len(one_segment_preds) * thresh)
            ensembled_pred_list.append(ensembled_pred)
        orig_mat_list['preictal'] = ensembled_pred_list
        return orig_mat_list

    # preds to csv
    # sub_df = pd.read_csv('../output/sampleSubmission.csv')
    sub_df = pd.read_csv(args.sub_path, engine='python')
    thresh = args.thresh  # 1の割合がthreshを超えたら1と判断
    pred_df = ensemble_preds(pred_list, path_list, sub_df, thresh)
    sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
    sub_df.to_csv(args.sub_path, index=False)

    # TODO print prediction


if __name__ == '__main__':
    args = test_args().parse_args()
    init_seed(args)
    device = init_device(args)

    test(args)
