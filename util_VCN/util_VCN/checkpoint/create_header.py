import numpy as np


def create_header_clas(mode, log_val_only, header_train=None, header_eval=None):
    if mode == 'train':
        if header_train is None:
            if log_val_only:
                header_train = ['itr', 'Loss/train', 'Loss/val', 'Accuracy/val']
            else:
                header_train = ['itr', 'Loss/train', 'Loss/val', 'Accuracy/val', 'Loss/test', 'Accuracy/test']
        multi_epo = np.asarray(header_train)
    else:
        if header_eval is None:
            header_eval = ['loss', 'accuracy', 'precision', 'recall', ' F1-score']
        multi_epo = np.asarray(header_eval)

    return multi_epo.reshape(1,-1)


def create_header_seg(mode, log_val_only, header_train=None, header_eval=None):
    if mode == 'train':
        if header_train is None:
            if log_val_only:
                header_train = ['itr', 'Loss/train', 'Loss/val', 'Dice/val']
            else:
                header_train = ['itr', 'Loss/train', 'Loss/val', 'Dice/val', 'Loss/test', 'Dice/test']
        multi_epo = np.asarray(header_train)
    else:
        if header_eval is None:
            header_eval = ['loss', 'dice', 'iou', 'precision', 'recall']
        multi_epo = np.asarray(header_eval)

    return multi_epo.reshape(1,-1)


def create_header_regres(mode, log_val_only, header_train=None, header_eval=None):
    if mode == 'train':
        if header_train is None:
            if log_val_only:
                header_train = ['itr', 'Loss/train', 'Loss_MSE/val', 'l1/val']
            else:
                header_train = ['itr', 'Loss/train', 'Loss_MSE/val', 'l1/val', 'Loss_MSE/test', 'l1/test']
        multi_epo = np.asarray(header_train)
    else:
        if header_eval is None:
            header_eval = ['Loss_MSE', 'l1', 'rmse', 'r2']
        multi_epo = np.asarray(header_eval)

    return multi_epo.reshape(1,-1)