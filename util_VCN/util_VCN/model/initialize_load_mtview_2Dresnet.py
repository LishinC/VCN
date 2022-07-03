import torch
# # The following is set when generating and saving the initialization to be used by all model variations. Comment these for main runs to avoid conflict.
# torch.manual_seed(0)
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(0)
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from util_VCN.model.mtview_parts.ConcateMerge import Merge2Dresnet as Merge
from util_VCN.model.mtview_parts.resnet_mod import ResNet


class Head(nn.Module):
    def __init__(self, embed_dim=128, c=2048, **kwargs):
        super().__init__()
        self.fc = nn.Linear(c, embed_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)             # [B,c=2048,1,1] -> [B, c=2048]
        x = self.fc(x)                      # [B,embed_dim=128]
        x = F.normalize(x, p=2, dim=1)      # L2 normalization, such that torch.norm(x[i,:])=1 for all i
        return x


def initialize_load_model(mode, model_path='scratch', required_views=['A4C', 'A2C'],
                          submodule_path=None, freeze_submodule=False, Merge=Merge, device="cuda", Resnet=ResNet,
                          Head=Head, warm_start_path=None, **kwargs):

    S, H = [], [] #S for submodules (one branch for each view). H for heads (one head for each branch)
    for i, view in enumerate(required_views):
        submodule = Resnet(**kwargs)
        if freeze_submodule:
            for param in submodule.parameters():
                param.requires_grad = False
        if (mode=='train') & (submodule_path is not None):
            submodule.load_state_dict(torch.load(submodule_path[i]))
        # submodule.load_state_dict(torch.load('../../../cvon_vol_regres/model/resnet50_sgview_init.pth')['model'])  #Temporary code for initializating all branches identically
        submodule = nn.Sequential(*list(submodule.children())[:-1])     # Removing the final linear layer, output size [B, 2048, 1, 1]
        S.append(submodule)

        head = Head(**kwargs)
        # head.load_state_dict(torch.load('../../../cvon_vol_regres/model/head_init.pth')['model'])  #Temporary code for initializating all branches identically
        H.append(head)

    model = Merge(S=S, H=H, **kwargs)
    if (mode == 'train') & (warm_start_path is not None):
        model.load_state_dict(torch.load(warm_start_path)['model'])
        # TODO: This is a special loading where the checkpoint contains both model weights and optimizer weights
    elif (mode == 'test'):
        #TODO: Updated checkpoint_train such that the pth file contains both model weights and optimizer weights
        model.load_state_dict(torch.load(model_path)['model'])

    model.to(device)
    param = model.parameters()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    return model, param


# if __name__ == '__main__':
#     """
#     As compared to the last version, a linear head and L2 normalization before contrastive loss is added.
#     The resnet is a 2D one (instead of 3D) identical to resnet50 except the first and last layer
#
#     Corresponding update in main:
#     three model outputs to unpack. x_feature and x_embed have no redundant dimensions in the end.
#     The input X remain 3D, with shape [B, in_channel=1, 2(A4C&A2C), x, y]
#     """
#     #TODO: note that resnet_num_frame=1 in this case since each branch process only one view
#     B=5
#     in_channel, out_channel = 1, 1
#     kwargs = {'in_channel': in_channel, 'out_channel': 1}
#     model, p = initialize_load_model('train',**kwargs)
#
#     # # # Save common initialization Head
#     # # model = Head()
#     # # torch.save({
#     # #     'model': model.state_dict(),
#     # # }, '../../../cvon_vol_regres/model/head_init.pth')
#     #
#     # # Save common initialization for all contrastive loss variations
#     # # opt = torch.optim.SGD(p, lr=1e-4, weight_decay=1e-8)
#     # opt = torch.optim.Adam(p, lr=1e-4, weight_decay=1e-8)
#     # torch.save({
#     #     'model': model.state_dict(),
#     #     'opt': opt.state_dict()
#     # }, '../../../cvon_vol_regres/model/resnet50_mtview_init.pth')
#
#     # Try forward
#     X = torch.randn([B, in_channel, 2, 256, 256]).to("cuda")
#     x_feature, x_embed, x = model(X)
#     print(x_feature.shape, x_embed.shape, x.shape) # [B, 2048, num_view=2], [B, embed_dim=128, num_view=2], [B, out_channel=1]
#
#
#     # from torchsummary import summary
#     # summary(model, (1, 2, 256, 256))