PHASES = ['train', 'val', 'test']
LABELS_2 = {'interictal': 0, 'preictal': 1}
LABELS_3 = {'interictal': 0, 'preictal': 1, 'ictal': 2}

# chb-mit
CHANNEL_CHANGED_PATIENTS = ['chb04', 'chb09', 'chb11', 'chb12', 'chb13', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19'] + ['chb07']
TIME_NOT_LISTED_PATIENTS = ['chb24']
CHANNELS_28 = ['chb14', 'chb21', 'chb20', 'chb22']
INNER_SUBJECTS = list(set([f'chb{i:02}' for i in range(1, 25)]) - set(CHANNEL_CHANGED_PATIENTS) -
                      set(TIME_NOT_LISTED_PATIENTS))
SUBJECTS = list(set(INNER_SUBJECTS) - set(CHANNELS_28))
INNER_SUBJECTS.sort()
SUBJECTS.sort()
