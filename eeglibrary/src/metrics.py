import torch


def recall_rate(y_true, y_pred):
    y_true, y_pred = y_true.float(), y_pred.float()
    # le(0.0) makes bits reverse
    tp = torch.dot(y_true, y_pred).sum()
    fn = torch.dot(y_true, y_pred.le(0).float()).sum()
    if torch.add(tp, fn) == 0:
        return torch.zeros(1)
    return tp.div(torch.add(tp, fn))


def false_detection_rate(y_true, y_pred):
    y_true, y_pred = y_true.float(), y_pred.float()
    # fdr = 1 - specifity
    fp = torch.dot(y_true.le(0).float(), y_pred).sum()
    tn = torch.dot(y_true.le(0).float(), y_pred.le(0).float()).sum()
    if torch.add(fp, tn) == 0:
        return torch.zeros(1)
    return fp.div(torch.add(fp, tn))
