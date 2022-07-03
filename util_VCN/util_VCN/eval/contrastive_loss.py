import torch
from torch import nn
# from torch.autograd import Variable
# from torch import einsum
# import numpy as np
from itertools import combinations


class ContrastiveLoss(nn.Module):
    """
    Phase contrastive loss / Volume contrastive loss if custom_margin assigned
    accept model 2 outputs (ED & ES), both of shape [B, embed_dim=128, 2].
    The 2nd-dimension contains the N-dimensional vector embeddings for frames,
    namely A4C ED, A2C ED in the first array; A4C ES, A2C ES in the second array.
    The loss encourage similar embeddings for the same phase, regardless of view.
    The positive pairs are the the ES pair and the ED pair.
    The negative pairs are the A4C pair and the A2C pair.
    """
    def __init__(self, default_margin=0.25):
        super().__init__()

        self.default_margin = default_margin

    def distance(self, a, b, channel_dim=1):
        diff = torch.abs(a - b)
        return torch.pow(diff, 2).sum(dim=channel_dim) #[B, num_pairs=1]
    
    def forward(self, outputED, outputES, custom_margin=None):
        # assert  outputED.shape[1:] == (128,2)
        if custom_margin is None:
            custom_margin = self.default_margin
        else:
            custom_margin[custom_margin>1] = 1      #For volume contrastive loss, constrain margin to less than 1
        d_positive, d_negative = 0, 0

        for embed in [outputED, outputES]:
            d_positive += self.distance(   embed[:, :, 0],    embed[:, :, 1])

        for i in range(2):
            d_negative += self.distance(outputED[:, :, i], outputES[:, :, i])
        loss = torch.clamp(custom_margin + d_positive - d_negative, min=0.0).mean()
        # print('d_positive d_negative', d_positive, d_negative)
        return loss


class ContrastiveLoss_interSubject(ContrastiveLoss):
    """
    Volumn contrastive loss & inter-subject contarsting.
    The positive example is a image from the same phase with a similar volume.
    The negative example is the image of the counter phase from the same subject.
    Previous intra-subject contrast: Despite coming from different 'view', if the volume is similar, then the embedding should be similar.
    Now, inter-subject contrast: Despite coming from different 'subject', if the volume is similar, then the embedding should be similar.
    """

    def __init__(self, default_margin=0.25):
        super().__init__(default_margin)

    def forward(self, anchor, pos, neg, Vdiff=None):
        # assert  anchor.shape == (128,2)
        # assert  pos.shape == (128,2)
        # assert  neg.shape == (128,2)
        d_positive, d_negative = 0, 0
        if Vdiff is None:
            Vdiff = self.default_margin
        elif Vdiff > 1:
            Vdiff = 1

        # Loop over 2 views with i.
        # The inputs have no batch dimension after indexing in main, thus summing over dimension 0 instead of 1
        for i in range(2):
            d_positive += self.distance(anchor[:, i], pos[:, i], channel_dim=0)
            d_negative += self.distance(anchor[:, i], neg[:, i], channel_dim=0)
        loss = torch.clamp(Vdiff + d_positive - d_negative, min=0.0).mean()
        # print('d_positive d_negative', d_positive, d_negative)
        return loss


class ContrastiveLoss_interSubject_sgview(ContrastiveLoss):

    def __init__(self, default_margin=0.25):
        super().__init__(default_margin)

    def forward(self, anchor, pos, neg, Vdiff=None):
        # assert  anchor.shape == (2048,)
        # assert  pos.shape == (2048,)
        # assert  neg.shape == (2048,)
        d_positive, d_negative = 0, 0
        if Vdiff is None:
            Vdiff = self.default_margin
        elif Vdiff > 1:
            Vdiff = 1

        # Loop over 2 views with i.
        # The inputs have no batch dimension after indexing in main, thus summing over dimension 0 instead of 1
        d_positive += self.distance(anchor, pos, channel_dim=0)
        d_negative += self.distance(anchor, neg, channel_dim=0)
        loss = torch.clamp(Vdiff + d_positive - d_negative, min=0.0).mean()
        # print('d_positive d_negative', d_positive, d_negative)
        return loss


class Distance_construct(nn.Module):
    """
    Note: Seemly calculating the loss based on Euclidean distance lead to unstable training because of taking square root.
    Network would output embeddings consists of NaNs after 1 update. Stick to Euclidean distance square (without
    taking sqrt) just like triplet loss.

    Learn the distance between the anchor sample and some neighbor(s) with similar volume / all samples in the batch.
    embed: [B, 2048]
    idx_anchor: integer indicating which sample in the batch is to be the anchor
    idx_neibor: the nearest neighbor(s) to be compare embedding distance with the anchor. If not indicated, all samples
                in the batch are compared.
    distances: [B,]
    """

    def __init__(self):
        super().__init__()

    def forward(self, embed, idx_anchor, idx_neibor, volume_scaled):
        # assert  embed.shape[1:] == (2048,)
        # distances[distances > 1] = 1  # For volume contrastive loss, constrain margin to less than 1

        if idx_anchor is None:
            loss = 0
            for idx_anchor in range(len(embed)):
                dist_vol = torch.abs(volume_scaled - volume_scaled[idx_anchor, :]).reshape(-1, )
                embed = embed - embed[idx_anchor, :]            # [B, 2048]
                dist_embed = torch.pow(embed, 2).sum(dim=1)     # [B,]
                loss += torch.abs(dist_embed - dist_vol).mean() # [B,] -> []
        else:
            dist_vol = torch.abs(volume_scaled - volume_scaled[idx_anchor,:]).reshape(-1,)
            embed = embed - embed[idx_anchor,:]                     # [B, 2048]
            dist_embed = torch.pow(embed, 2).sum(dim=1)             # [B,]
            loss = torch.abs(dist_embed - dist_vol)                 # [B,]
            if idx_neibor is not None:  #If neighbor(s) are provided, take into account only the distance between anchor and those
                idx_keep = [False] * len(embed)
                for a in idx_neibor:
                    idx_keep[a] = True
                loss = loss[idx_keep]
            loss = loss.mean()
        return loss


# if __name__ == '__main__':
#     d = Distance_construct()
#     embed = torch.tensor([[1,2,4,5],[1,1,1,1],[2,3,4,5]]).float()
#     loss = d(embed, 1, torch.tensor([5,0,5]))
#     print(loss)
# fr = torch.tensor([8, 4, 1, 2, 12])
# torch.topk(fr, 3).indices
# torch.kthvalue(fr, 3)[1]