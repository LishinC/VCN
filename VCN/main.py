import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
np.random.seed(0)
from util_VCN.backbone.backbone_loops import backbone_loops
from util_VCN.loader.CAMUS.loader_4frame import create_dataloader
from util_VCN.eval.eval import one_epoch_avg_regres_multi_losses
from util_VCN.model.initialize_load_sgview_2Dresnet import initialize_load_model as init_sb
from util_VCN.model.initialize_load_mtview_2Dresnet import initialize_load_model as init_mt
import torch.nn.functional as F
from util_VCN.eval.contrastive_loss import ContrastiveLoss, ContrastiveLoss_interSubject, ContrastiveLoss_interSubject_sgview


class Head_noProjection(nn.Module):
    def __init__(self, embed_dim=128, c=2048, **kwargs):
        super().__init__()
        self.fc = nn.Linear(c, embed_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)             # [B,c=2048,1,1] -> [B, c=2048]
        x = F.normalize(x, p=2, dim=1)      # L2 normalization, such that torch.norm(x[i,:])=1 for all i
        return x


def unpack_toDevice(batch, device):
    X, Y, EDV, ESV, ID = batch
    X = X.to(device)
    EDV = EDV.to(device)
    ESV = ESV.to(device)
    return X, Y, EDV, ESV, ID


def batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, log1, log2, return_one_batch, which_phase):
    if return_one_batch:
        if which_phase == 'ED':
            y_pred = out_logitED.item()
            y_true = EDV.item()
        if which_phase == 'ES':
            y_pred = out_logitES.item()
            y_true = ESV.item()
        one_batch = [ID[0], log1, log2, y_true, y_pred]
        return [], one_batch
    else:
        return loss, []


def calculate_regres(criterion, out_logitED, out_logitES, EDV, ESV):
    lossED = criterion(out_logitED, EDV)
    lossES = criterion(out_logitES, ESV)
    return lossED + lossES


def calculate_intra(criterion, out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, varying_margin=True):
    Vdiff = (EDV-ESV)/Vdiff_scaling
    if varying_margin:
        loss_intra = criterion(out_embedED, out_embedES, Vdiff.reshape(-1))
    else:
        loss_intra = criterion(out_embedED, out_embedES)
    return loss_intra


def calculate_inter(criterion, out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch, varying_margin=True):
    loss_inter = 0
    cnt=[]
    if not return_one_batch:        # When inference on single sample don't calculate inter_subject loss
        BATCH_SIZE = len(EDV)
        for idx_anc in range(BATCH_SIZE):
            for p, V, embed, embed_counter in zip([0,1], [EDV, ESV], [out_embedED, out_embedES], [out_embedES, out_embedED]):
                # For each anchor (idx_anc in the batch), try to find in the batch another sample (idx_pos),
                # which has similar volume (V) at the same phase, to be the positive example
                # The counter phase of the anchor sample is the negative example.
                inter_sub_diff = abs(V - V[idx_anc]).reshape(-1,)
                intra_sub_diff = abs(EDV-ESV)[idx_anc]                  #The ED-ES difference
                pos_candidates_bool = inter_sub_diff < intra_sub_diff   #Find valid positive examples. Exclude the anchor itself.
                pos_candidates_bool[idx_anc] = False
                if pos_candidates_bool.sum() == 0:
                    cnt.append(0)
                else:
                    idx_pos_candidates = torch.nonzero(pos_candidates_bool)
                    idx_pos = idx_pos_candidates[torch.randperm(len(idx_pos_candidates))[0]][0]
                    Vdiff = (intra_sub_diff - inter_sub_diff[idx_pos])/Vdiff_scaling
                    assert Vdiff > 0
                    if varying_margin:
                        loss_inter += criterion(anchor=embed[idx_anc,...],
                                                pos=embed[idx_pos,...], neg=embed_counter[idx_anc,...], Vdiff=Vdiff)
                    else:
                        loss_inter += criterion(anchor=embed[idx_anc,...],
                                                pos=embed[idx_pos,...], neg=embed_counter[idx_anc,...])
                    cnt.append(1)

        loss_inter = loss_inter/2                           # Repeated for both phases, hence twice larger than TCN loss

    return loss_inter


def A2C(batch, model, device, return_one_batch, criterion=nn.MSELoss(), which_phase='ED', **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_logitED = model(X[:,:,2,:,:]) #Use only A2C ED
    _, out_logitES = model(X[:,:,3,:,:]) #Use only A2C ES

    loss = calculate_regres(criterion, out_logitED, out_logitES, EDV, ESV)

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss.item(), 0, return_one_batch, which_phase)


def A4C(batch, model, device, return_one_batch, criterion=nn.MSELoss(), which_phase='ED', **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_logitED = model(X[:, :, 0, :, :])  # Use only A4C ED
    _, out_logitES = model(X[:, :, 1, :, :])  # Use only A4C ES

    loss = calculate_regres(criterion, out_logitED, out_logitES, EDV, ESV)

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss.item(), 0, return_one_batch, which_phase)


def A2C_inter(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss_interSubject_sgview()],
              BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    out_featureED, out_logitED = model(X[:,:,2,:,:]) #Use only A2C ED
    out_featureES, out_logitES = model(X[:,:,3,:,:]) #Use only A2C ES

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Inter-subject loss
    out_embedED, out_embedES = F.normalize(out_featureED, p=2, dim=1), F.normalize(out_featureES, p=2, dim=1)
    loss_inter = calculate_inter(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch)

    loss = loss_regres + BETA*loss_inter

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), 0, return_one_batch, which_phase)


def A4C_inter(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss_interSubject_sgview()],
              BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    out_featureED, out_logitED = model(X[:,:,0,:,:]) #Use only A4C ED
    out_featureES, out_logitES = model(X[:,:,1,:,:]) #Use only A4C ES

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Inter-subject loss
    out_embedED, out_embedES = F.normalize(out_featureED, p=2, dim=1), F.normalize(out_featureES, p=2, dim=1)
    loss_inter = calculate_inter(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch)

    loss = loss_regres + BETA*loss_inter

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), 0, return_one_batch, which_phase)


def SimpleEarly(batch, model, device, return_one_batch, criterion=nn.MSELoss(), which_phase='ED', **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_logitED = model(X[:,0,[0,2],:,:])      #Use ED, two views concatenated at the channel dimension
    _, out_logitES = model(X[:,0,[1,3],:,:])      #Use ES, two views concatenated at the channel dimension

    loss = calculate_regres(criterion, out_logitED, out_logitES, EDV, ESV)

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss.item(), 0, return_one_batch, which_phase)


def SimpleLate(batch, model, device, return_one_batch, criterion=nn.MSELoss(), which_phase='ED', **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss = calculate_regres(criterion, out_logitED, out_logitES, EDV, ESV)

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss.item(), 0, return_one_batch, which_phase)


def VCN(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss(), ContrastiveLoss_interSubject()],
       ALPHA = 1000, BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Intra-subject loss
    loss_intra = calculate_intra(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling)

    # Inter-subject loss
    loss_inter = calculate_inter(criterion[2], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch)

    loss = loss_regres + ALPHA*loss_intra + BETA*loss_inter

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), loss_intra.item(), return_one_batch, which_phase)


def VCN_fixed(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss(), ContrastiveLoss_interSubject()],
              ALPHA = 1000, BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Intra-subject loss
    loss_intra = calculate_intra(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, varying_margin=False)

    # Inter-subject loss
    loss_inter = calculate_inter(criterion[2], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch, varying_margin=False)

    loss = loss_regres + ALPHA*loss_intra + BETA*loss_inter

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), loss_intra.item(), return_one_batch, which_phase)


def Intra(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss()],
          ALPHA = 1000, BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Intra-subject loss
    loss_intra = calculate_intra(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling)

    loss = loss_regres + ALPHA*loss_intra

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), loss_intra.item(), return_one_batch, which_phase)


def Intra_fixed(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss()],
                ALPHA = 1000, BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Intra-subject loss
    loss_intra = calculate_intra(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, varying_margin=False)

    loss = loss_regres + ALPHA*loss_intra

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), loss_intra.item(), return_one_batch, which_phase)


def Inter(batch, model, device, return_one_batch, criterion=[nn.MSELoss(), ContrastiveLoss_interSubject()],
          ALPHA = 1000, BETA=1000, which_phase='ED', Vdiff_scaling=200, **kwargs):
    X, Y, EDV, ESV, ID = unpack_toDevice(batch, device)

    _, out_embedED, out_logitED = model(X[:,:,[0,2],:,:])   #Use ED, two views will be fed to 2D ResNet separatedly as single-channel image
    _, out_embedES, out_logitES = model(X[:,:,[1,3],:,:])   #Use ES, two views will be fed to 2D ResNet separatedly as single-channel image

    # Regression loss
    loss_regres = calculate_regres(criterion[0], out_logitED, out_logitES, EDV, ESV)

    # Inter-subject loss
    loss_inter = calculate_inter(criterion[1], out_embedED, out_embedES, EDV, ESV, Vdiff_scaling, return_one_batch)

    loss = loss_regres + BETA*loss_inter

    return batch_output(out_logitED, out_logitES, EDV, ESV, ID, loss, loss_regres.item(), 0, return_one_batch, which_phase)


if __name__ == '__main__':
    fold_itr = range(5)

    root_list = ['0001_A2COnly/', '0002_A4COnly/', '0009_A2C_inter/', '0010_A4C_inter/', '0003_SimpleEarly/',
                 '0004_SimpleLate/', '0005_VCN/', '0006_intra/', '0007_inter/', '0008_intra_FixMargin/',
                 '0011_VCN_LinearHead/', '0012_intra_LinearHead/', '0013_inter_LinearHead/']
    fwd_list = [A2C, A4C, A2C_inter, A4C_inter, SimpleEarly,
                SimpleLate, VCN, Intra, Inter, Intra_fixed,
                VCN, Intra, Inter]
    init_list = [init_sb]*5 + [init_mt]*8
    ws_list = ['./model/resnet50_sgview_init.pth']*5 + ['./model/resnet50_mtview_init.pth']*9

    exp_kwargs = []
    assert len(root_list) == len(fwd_list) == len(init_list) == len(ws_list)
    for k in fold_itr:
        kwargs_data = {'batch_size': 16, 'foldk': k, 'split_folder': '5fold/'}
        for j in range(len(root_list)):
            save_folder = 'model/fold'+str(k)+'/' + root_list[j]
            kwargs = {'save_folder': save_folder, 'forward': fwd_list[j], 'task': 'regres',
                      'initialize_load_model': init_list[j], 'one_epoch_avg': one_epoch_avg_regres_multi_losses,
                      'header_train': ['itr', 'Loss/train', 'Loss_MSE/val', 'Loss_TCN/val', 'l1/val'],
                      'header_eval': ['Loss_MSE/val', 'Loss_TCN/val', 'l1', 'rmse', 'r2'], 'warm_start_path': ws_list[j]}
            if j <= 9: kwargs['Head'] = Head_noProjection
            if j == 4:
                kwargs['in_channel'] = 2
                kwargs['early_fusion_in_channel_2'] = True
            exp_kwargs.append(kwargs)


        backbone_loops(exp_kwargs, create_dataloader, range(1600), **kwargs_data)
        for kw in exp_kwargs: kw['which_phase'] = 'ES'
        backbone_loops(exp_kwargs, create_dataloader, range(1600), workflow='test', subfolder='test_ES', **kwargs_data)
