from datetime import datetime
from tqdm import tqdm
import sys
from util_VCN.backbone.Experiment import Experiment


def backbone_loops(exp_kwargs, create_dataloader, epo_iter, workflow='complete', subfolder='default', **kwargs_data):
    try:
        experiments = [Experiment(**kwargs) for kwargs in exp_kwargs]
        n = datetime.now()
        assert workflow in ['complete', 'train', 'test']
        dataloader_train = create_dataloader(mode='train', **kwargs_data)
        dataloader_val   = create_dataloader(mode='val', **kwargs_data)
        dataloader_test  = create_dataloader(mode='test', **kwargs_data)

        ## [Training]
        if (workflow == 'complete') | (workflow == 'train'):
            for exp in experiments: exp.train_initialization()

            for epo in tqdm(epo_iter, ncols=0):
                for exp in experiments: exp.create_empty_logging_list()
                for i, batch in enumerate(dataloader_train):
                    wt = (datetime.now() - n).total_seconds()
                    if wt>2: print('\n Batch loading waiting time ', wt)

                    for exp in experiments: exp.batch_train(batch)
                    n = datetime.now()

                is_last_update = epo == (epo_iter.stop - 1)
                for exp in experiments: exp.val(dataloader_val, logging_args=[epo, subfolder, is_last_update])
            for exp in experiments: exp.train_ending()

        ## [Testing]
        if (workflow == 'complete') | (workflow == 'test'):
            for exp in experiments: exp.test(dataloader_test, subfolder)

    except KeyboardInterrupt:
        ## [Test on current best if interrupted]
        print('Interrupted at epo ', epo)
        for exp in experiments: exp.test(dataloader_test, subfolder)
        sys.exit(0)