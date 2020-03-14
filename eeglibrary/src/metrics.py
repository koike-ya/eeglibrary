import torch
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from ml.src.metrics import Metric


def false_detection_rate(true, pred, numpy=True):
    if numpy:
        return np.dot(true == 0, pred == 1) / len(pred)

    fp = torch.dot(true.le(0).float(), pred.ge(1).float()).sum()
    return fp.div(len(pred)).item()


# def specificity(pred, true, numpy=False)
#     true, pred = true.float(), pred.float()
#     # fdr = 1 - specifity
#     fp = torch.dot(true.le(0).float(), pred).sum()
#     tn = torch.dot(true.le(0).float(), pred.le(0).float()).sum()
#     if torch.add(fp, tn) == 0:
#         return torch.zeros(1)
#     return fp.div(torch.add(fp, tn))
