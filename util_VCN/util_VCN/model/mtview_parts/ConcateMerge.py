import torch
import torch.nn as nn


class MergeR21D(nn.Module):
    def forward_concat(self, x):
        assert len(self.S) == len(x)

        out = []
        for i in range(len(x)):
            out.append(self.S[i](x[i]))  # Elements has shape [B,512,1,1,1]
        out = torch.stack(out, dim=1).squeeze(3).squeeze(3).squeeze(3)  # The output has shape [B,num_views,512]
        return out

    def __init__(self, S, out_channel):
        super().__init__()
        num_view = len(S)
        self.S = nn.ModuleList(S)
        self.fc = nn.Linear(512*num_view, out_channel)
        self.inorm = nn.InstanceNorm1d(num_view)

    def forward(self, x):
        concat_vec = self.forward_concat(x)
        concat_vec = self.inorm(concat_vec)     # convat_vec has shape [B, num_view, 512]. Each 512-dimensional vector would be standardized
        x = torch.flatten(concat_vec, 1)        # [B, num_view, 512] -> [B, num_view*512]
        x = self.fc(x)
        return concat_vec, x


class MergeRessNet(nn.Module):
    def forward_concat(self, x):
        assert len(self.S) == x.shape[2]

        out = []
        for i in range(len(self.S)):
            out.append(self.S[i](x[:,:,i,:,:].unsqueeze(2)))    # [B,1,1,256,256] -> [B,1024,num_frame=1,1,1]
        out = torch.cat(out, dim=2)                             # The output has shape [B,1024,num_views=2,1,1]
        return out

    def __init__(self, S, out_channel=1, c=[64, 128, 256, 512, 1024], **kwargs):
        super().__init__()
        num_view = len(S)
        self.S = nn.ModuleList(S)
        self.fc = nn.Linear(c[-1]*num_view, out_channel)

    def forward(self, x):
        concat_vec = self.forward_concat(x)
        x = torch.flatten(concat_vec, 1)        # [B,1024,num_views=2,1,1] -> [B, num_view*1024]
        x = self.fc(x)
        return concat_vec, x


class Merge2Dresnet(nn.Module):
    def forward_concat(self, x):
        assert len(self.S) == x.shape[2]

        feature_mtview, embed_mtview = [], []
        for i in range(len(self.S)):
            feature = self.S[i](x[:,:,i,:,:])               # [B,1,256,256] -> [B,2048,1,1]
            embed = self.H[i](feature)                      # [B,2048,1,1] -> [B, embed_dim=128]
            feature_mtview.append(feature)
            embed_mtview.append(embed)
        feature_mtview = torch.cat(feature_mtview, dim=2)   # feature_mtview has shape [B,2048,num_views=2,1]
        feature_mtview = feature_mtview[:,:,:,0]              # [B,2048,num_view=2]
        embed_mtview = torch.stack(embed_mtview, dim=2)     # [B,embed_dim=128,num_views=2]
        return feature_mtview, embed_mtview

    def __init__(self, S, H, out_channel=1, c=2048, **kwargs):
        super().__init__()
        num_view = len(S)
        self.S = nn.ModuleList(S)
        self.H = nn.ModuleList(H)
        self.fc = nn.Linear(c*num_view, out_channel)

    def forward(self, x):
        x_feature, x_embed = self.forward_concat(x)
        x = torch.flatten(x_feature, 1)            # [B,2048,num_views=2] -> [B, 2048*num_view]
        x = self.fc(x)
        return x_feature, x_embed, x