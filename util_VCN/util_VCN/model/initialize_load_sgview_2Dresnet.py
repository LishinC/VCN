import torch
# # The following is set when generating and saving the initialization to be used by all model variations. Comment these for main runs to avoid conflict.
# torch.manual_seed(0)
# # torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(0)
from util_VCN.model.mtview_parts.resnet_mod import ResNet
import torch.nn as nn


def initialize_load_model(mode, model_path='scratch', device="cuda", Resnet=ResNet, warm_start_path=None, early_fusion_in_channel_2=False, **kwargs): #TODO
    model = Resnet(**kwargs)
    if (mode=='train') & (warm_start_path is not None):
        if early_fusion_in_channel_2: #TODO
            model = Resnet() # This gives a ResNet with in_channel = 1
        model.load_state_dict(torch.load(warm_start_path)['model'])
        if early_fusion_in_channel_2: #TODO
            model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_path != 'scratch':
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
#     # Saving a random initialization that is to be used by all models as the start point
#     kwargs = {'in_channel': 1, 'out_channel': 1}
#     model , p = initialize_load_model('train', device="cuda", **kwargs)
#
#     # # Save common initialization for all contrastive loss variations
#     # opt = torch.optim.Adam(p, lr=1e-4, weight_decay=1e-8)
#     # torch.save({
#     #     'model': model.state_dict(),
#     #     'opt': opt.state_dict()
#     # }, '../../../cvon_vol_regres/model/resnet50_sgview_init.pth')
#
#     # Try forward
#     input_tensor = torch.randn(2, 1, 256, 256).cuda()
#     _, out = model(input_tensor)
#     print(out.shape)