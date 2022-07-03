import os
import numpy as np
import torch
from util_VCN.checkpoint.create_header import create_header_clas, create_header_seg, create_header_regres
from util_VCN.eval.eval import one_epoch_avg_clas, one_epoch_avg_seg, one_epoch_avg_regres


def checkpoint_test(one_epoch, model, save_folder, subfolder, task,
                    header_train=None, header_eval=None, one_epoch_avg=None, **kwargs):

    mode = 'test'
    create_header = globals()['create_header_'+task]
    if one_epoch_avg is None:
        one_epoch_avg = globals()['one_epoch_avg_'+task]
    if subfolder == 'default': subfolder = mode
    save_subfolder = save_folder + subfolder
    os.makedirs(save_subfolder, exist_ok=True)

    epo = find_epo_test(save_folder, subfolder, **kwargs)  # Here epo might actually be itr since the log might be per every update
    one_epoch_avg = one_epoch_avg(one_epoch)
    multi_epo = create_header(mode, None, header_train, header_eval)
    multi_epo = np.concatenate([multi_epo, one_epoch_avg], axis=0)
    np.savetxt(save_subfolder + '/prediction_' + str(epo) + '.csv', np.asarray(one_epoch), fmt='%s', delimiter=',')
    np.savetxt(save_subfolder + '/performance_' + str(epo) + '.csv', np.asarray(multi_epo), fmt='%s', delimiter=',')

    print('Epoch: ', epo, '| ', mode, ' | performance: ', one_epoch_avg, '\n')


def find_epo_test(save_folder, subfolder, **kwargs):
    # The columns of multi_epo are [itr, train_loss, val_loss_or_early_stop_metric, other_val_metrics_if_any]
    multi_epo = np.genfromtxt(save_folder + '/train/log.csv', dtype='str', delimiter=',')
    multi_epo = multi_epo[1:,2].astype('float')
    epo_test = np.argmin(multi_epo)
    min_loss = multi_epo[epo_test]
    os.makedirs(save_folder + '/' + subfolder, exist_ok=True)
    np.savetxt(save_folder + '/' + subfolder + '/minLoss_'+str(min_loss)+'.txt',[]) # Just to indicate val-loss. Empty file
    return epo_test