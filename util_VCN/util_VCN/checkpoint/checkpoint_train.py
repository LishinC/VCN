import os
import numpy as np
import torch
from util_VCN.checkpoint.create_header import create_header_clas, create_header_seg, create_header_regres
from util_VCN.eval.eval import one_epoch_avg_clas, one_epoch_avg_seg, one_epoch_avg_regres


def checkpoint_train(itr, one_epoch_train, one_epoch_val, one_epoch_test, model, opt, save_folder, subfolder, epo, is_first_update, is_last_update, writer, log_val_only,
               task, header_train=None, header_eval=None, one_epoch_avg=None, **kwargs):

    mode = 'train'
    create_header = globals()['create_header_'+task]
    if one_epoch_avg is None:
        one_epoch_avg = globals()['one_epoch_avg_'+task]
    save_subfolder = save_folder + mode     # From now on the trainign log is always stored in the folder "train"
    os.makedirs(save_subfolder, exist_ok=True)

    # TODO: make optional. The default setting is for clas. Modify this. Also uncomment the rename in if is_last_update.
    # log_individual_prediction(one_epoch_val, save_subfolder+'/individual_pred_tmp.csv', is_first_update)
    train_avg = np.mean(one_epoch_train)

    header = create_header(mode, log_val_only, header_train, header_eval)
    NUM_METRICS_TO_LOG = len(header[0]) - 2

    if log_val_only:
        # For clas, one_epoch contains appended [ID[0], loss.item(), y_true, y_pred]
        # one_epoch_avg returns numpy array of shape (1,-1) containing [loss, acc, prec, rec, f1]
        # Sor seg, one_epoch contains appended [ID[0], loss, 'dice', 'iou', 'precision', 'recall']
        # one_epoch_avg returns its average, with shape (1, -1)
        val_avg = one_epoch_avg(one_epoch_val)
        one_epoch_log = [itr, train_avg] + list(val_avg.reshape(-1,))[:NUM_METRICS_TO_LOG]
    else:
        val_avg = one_epoch_avg(one_epoch_val)
        test_avg = one_epoch_avg(one_epoch_test)
        one_epoch_log = [itr, train_avg] + list(val_avg.reshape(-1,))[:NUM_METRICS_TO_LOG/2] + list(test_avg.reshape(-1,))[:NUM_METRICS_TO_LOG/2]

    logging(one_epoch_log, header, writer, save_subfolder, is_first_update, log_val_only, model, opt, **kwargs)

    # if is_last_update:
    #     os.rename(save_subfolder + '/log_tmp.csv', save_subfolder + '/log_' + str(epo) + '.csv')
    #     # os.rename(save_subfolder + '/individual_pred_tmp.csv', save_subfolder + '/individual_pred_' + str(epo) + '.csv')

    print('Epoch: ', epo, '| training | performance: ', one_epoch_log, '\n')


def logging(one_epoch_log, header, writer, save_subfolder, is_first_update, log_val_only, model, opt, save_2models=False, save_model_each_epo=False, **kwargs):
    """
    1) Log performance to csv & tensorboard.
    2) Determine if has validation loss minimum.
    """
    
    def compare(one_epoch_log, multi_epo):
        current     = one_epoch_log[2]
        history_min = min(multi_epo[1:,2].astype('float'))
        if current < history_min:
            has_min_val = True
        else:
            has_min_val = False
        return has_min_val

    # Write to tensorboard
    itr = one_epoch_log[0]
    assert len(header[0]) == len(one_epoch_log)

    for i in range(1,len(header[0])):
        writer.add_scalar(header[0,i], one_epoch_log[i], itr)

    # Write to csv file & Save model if has val-loss minimum
    csv_name = save_subfolder+'/log.csv'
    if is_first_update:
        multi_epo = header
        has_min_val = True
    else:
        multi_epo = np.genfromtxt(csv_name, dtype='str', delimiter=',')
        has_min_val = compare(one_epoch_log, multi_epo)
    one_epoch_log = np.asarray(one_epoch_log).reshape(1, -1)
    multi_epo = np.concatenate([multi_epo, one_epoch_log], axis=0)
    np.savetxt(csv_name, np.asarray(multi_epo), fmt='%s', delimiter=',')
    if has_min_val | save_model_each_epo:
        print('save model')
        if save_2models:
            torch.save({
                'discriminator': model[0].state_dict(),
                'regressor': model[1].state_dict(),
                'opt0_state_dict': opt[0].state_dict(),
                'opt1_state_dict': opt[1].state_dict()
            }, save_subfolder + '/model_val_min.pth')
        else:
            # TODO: Updated such that the pth file contains both model weights and optimizer weights
            # torch.save(model.state_dict(), save_subfolder + '/model_val_min.pth')
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict()
            }, save_subfolder + '/model_val_min.pth')

# def log_individual_prediction(one_epoch, csv_name, is_first_update):
#     if is_first_update:
#         multi_epo = np.asarray(one_epoch)[:,2].reshape(1, -1)
#         # multi_epo = np.asarray(one_epoch)[:,0].reshape(1, -1)     # For seg
#     else:
#         multi_epo = np.genfromtxt(csv_name, dtype='str', delimiter=',')
#     if multi_epo.ndim == 1: multi_epo = multi_epo.reshape(1, -1)
#     one_epoch = np.asarray(one_epoch)[:,4].reshape(1, -1)
#     # one_epoch = np.asarray(one_epoch)[:,1].reshape(1, -1)         # Fpr seg
#     multi_epo = np.concatenate([multi_epo, one_epoch],axis=0)
#     np.savetxt(csv_name, np.asarray(multi_epo), fmt='%s', delimiter=',')
