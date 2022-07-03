import torch
from util_VCN.model.initialize_load_sgview_2Dresnet import initialize_load_model
from util_VCN.checkpoint.checkpoint_train import checkpoint_train
from util_VCN.checkpoint.checkpoint_test  import checkpoint_test
from torch.utils.tensorboard import SummaryWriter


class Experiment():
    def __init__(self, save_folder, forward, task='clas', initialize_load_model=initialize_load_model,
                 checkpoint_train=checkpoint_train, checkpoint_test = checkpoint_test,
                 optimizer=torch.optim.Adam, lr=1e-4, wd=1e-8, **kwargs):

        assert task in ['clas', 'seg', 'regres']

        """
        The Experiment class carries all the current status about an experiment (model, optimizer), and have
        class methods (training, validating, testing, logging) which relies on the data given by backbone_loops
        """

        # Variables
        self.save_folder = save_folder
        self.task = task
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.kwargs = kwargs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Functions
        self.forward = forward
        self.initialize_load_model = initialize_load_model      # For loading checkpoint model during testing
        self.checkpoint_train = checkpoint_train
        self.checkpoint_test = checkpoint_test

    def train_initialization(self):
        """
        Initialize objects needed for training: model(training mode), optimizer, tensorboard writer
        """
        self.model, param = self.initialize_load_model(mode='train', device=self.device, **self.kwargs)
        self.opt = self.optimizer(param, lr=self.lr, weight_decay=self.wd)
        self.writer = SummaryWriter(self.save_folder)

    def create_empty_logging_list(self):
        """
        Executed at the beginning of each epoch. Create an empty list for logging training loss for each batch
        """
        self.one_epoch_train = []

    def train_ending(self):
        """
        Close tensorboard at the end of training
        """
        self.writer.flush()
        self.writer.close()

    def batch_train(self, batch):
        """
        Given a batch, run forward, obtain the loss, then run backward and update model weights.
        """
        self.opt.zero_grad()
        loss, _ = self.forward(batch, self.model, self.device, return_one_batch=False, **self.kwargs)
        self.one_epoch_train.append(loss.item())
        loss.backward()
        self.opt.step()

    def whole_eval(self, dataloader):
        """
        Given dataloader_val or dataloader_test, run inference on all the samples and return the performance list
        """
        self.model.eval()
        with torch.no_grad():
            one_epoch = []
            for i, batch in enumerate(dataloader):
                _, one_batch = self.forward(batch, self.model, self.device, return_one_batch=True, **self.kwargs)
                one_epoch.append(one_batch)
        self.model.train()

        return one_epoch

    def val(self, dataloader_val, logging_args):
        """
        Run whole_eval on the validation set and run checkpoint_train

        # In the old versions gradient accumulation was possible and eval_per_iter was an option. In the updated version
        # only log at the end of each epo, so give tensorboard "epo" instead of "itr"
        """
        one_epoch_val = self.whole_eval(dataloader_val)
        one_epoch_test = []

        epo, subfolder, is_last_update = logging_args
        is_first_update = epo == 0
        self.checkpoint_train(epo, self.one_epoch_train, one_epoch_val, one_epoch_test, self.model, self.opt,
                              self.save_folder, subfolder, epo, is_first_update, is_last_update,
                              self.writer, True, self.task, **self.kwargs)

    def test(self, dataloader_test, subfolder):
        """
        Run whole_eval on the test set and run checkpoint_test
        """
        self.model, _ = self.initialize_load_model(mode='test', model_path=self.save_folder+'train/model_val_min.pth',
                                                   device=self.device, **self.kwargs)
        one_epoch_test = self.whole_eval(dataloader_test)
        self.checkpoint_test(one_epoch_test, self.model, self.save_folder, subfolder, self.task, **self.kwargs)