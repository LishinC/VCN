import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def performance_seg(pred, true):
    overlap = pred & true              # TP
    union = pred | true                # TP + FN + FP
    misclassified = overlap != union    # FN + FP
    
    FP = (misclassified & pred).sum()
    FN = (misclassified & true).sum()
    TP = overlap.sum()
    TN = (~pred & ~true).sum()
    UN = union.sum()

    if UN == 0:
        dice = iou = precision = recall = accuracy = 1
    elif TP == 0:
        dice = iou = precision = recall = accuracy = 0
    else:
        dice = (TP*2) / (UN + TP)
        iou = TP / UN
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (UN + TN)
    
    return [dice, iou, precision, recall]


def one_epoch_avg_seg(one_epoch):
    one_epoch = np.asarray(one_epoch)[:, 1:].astype(np.float)
    avg = np.mean(one_epoch, axis=0, keepdims=True)
    return avg


def performance_clas(Y_true, Y_pred):
    acc = accuracy_score(Y_true, Y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average='weighted')
    return [acc, prec, rec, f1]


def one_epoch_avg_clas(one_epoch):
    one_epoch = np.asarray(one_epoch)[:,:4]
    Y_true = one_epoch[:, 2].astype(np.float)
    Y_pred = one_epoch[:, 3].astype(np.float)
    loss = one_epoch[:, 1].astype(np.float).mean()

    acc, prec, rec, f1 = performance_clas(Y_true, Y_pred)
    return np.asarray([loss, acc, prec, rec, f1]).reshape(1,-1)


def one_epoch_avg_clas_two_loss(one_epoch):
    one_epoch = np.asarray(one_epoch)[:,:5]
    Y_true = one_epoch[:, 3].astype(np.float)
    Y_pred = one_epoch[:, 4].astype(np.float)
    loss1 = one_epoch[:, 1].astype(np.float).mean()
    loss2 = one_epoch[:, 2].astype(np.float).mean()

    acc, prec, rec, f1 = performance_clas(Y_true, Y_pred)
    return np.asarray([loss1, loss2, acc, prec, rec, f1]).reshape(1,-1)


def performance_regres(Y_true, Y_pred):
    l1 = mean_absolute_error(Y_true, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
    r2 = r2_score(Y_true, Y_pred)
    return [l1, rmse, r2]


def one_epoch_avg_regres(one_epoch):
    one_epoch = np.asarray(one_epoch)[:,:4]
    Y_true = one_epoch[:, 2].astype(np.float)
    Y_pred = one_epoch[:, 3].astype(np.float)
    loss = one_epoch[:, 1].astype(np.float).mean()

    l1, rmse, r2 = performance_regres(Y_true, Y_pred)
    return np.asarray([loss, l1, rmse, r2]).reshape(1,-1)


def one_epoch_avg_regres_multi_losses(one_epoch):
    """
    Requires one_epoch's columns to be [ID, loss_main, ..., loss_n, y_true, y_pred]
    """
    one_epoch = np.asarray(one_epoch)
    Y_true = one_epoch[:, -2].astype(np.float)
    Y_pred = one_epoch[:, -1].astype(np.float)

    losses = []
    num_losses = one_epoch.shape[1] - 3
    for i in range(num_losses):
        losses.append(one_epoch[:, i+1].astype(np.float).mean())

    l1, rmse, r2 = performance_regres(Y_true, Y_pred)
    metrics = losses + [l1, rmse, r2]
    return np.asarray(metrics).reshape(1,-1)
