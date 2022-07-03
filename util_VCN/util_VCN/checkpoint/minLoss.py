import os
import numpy as np


def find_epo_test_from_val_log(save_folder, epo_iter, val_folder='val', test_folder='test', del_after_val=True, **kwargs):
    # Validation log must start with loss
    # Allows assigning special validation folder with kwargs
    csv_name = save_folder + val_folder + '/log_'+str(epo_iter.stop-1)+'.csv'
    multi_epo = np.genfromtxt(csv_name, dtype='str', delimiter=',')
    multi_epo = multi_epo[1:,0].astype('float')
    epo_test = np.argmin(multi_epo)
    min_loss = multi_epo[epo_test]
    np.savetxt(save_folder + test_folder +'/minLoss_'+str(min_loss)+'.txt',[])
    if del_after_val:
        for epo in epo_iter:
            if epo != epo_test:
                os.remove(save_folder + 'train/model_' + str(epo) + '.pth')
    return epo_test