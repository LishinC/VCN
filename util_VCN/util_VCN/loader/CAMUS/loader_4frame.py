import numpy as np
# from glob2 import glob
import torch
from torch.utils.data.dataset import Dataset
# import random
from random import randint
import cv2
from numpy.random import uniform, normal
from skimage.transform import resize
import skimage.io as io
import albumentations as A


def create_transform(aug, num_frame=2, frame_size=256, aug_shift_limit=0.1, aug_rotate_deg=15, aug_scale_limit=0, aug_hflip=False, aug_elastic=False, **kwargs):
    # Create dictionary of list of additional targets: {'image1':'image',  'image2':'image', ...}.
    # Specify num_frame and the following code don't have to be changed
    K = ['image' + str(i) for i in range(1, num_frame, 1)]
    V = ['image'] * (num_frame-1)
    additional_targets = {key: value for (key, value) in zip(K, V)}

    # Include augmentations if aug=True, otherwise simply pad to a square image and rescale to 256
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5) if aug&aug_hflip else None,
            A.PadIfNeeded(1232, 1232, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.ElasticTransform(alpha=1500, border_mode=cv2.BORDER_CONSTANT, value=0, p=1) if aug&aug_elastic else None,
            A.ShiftScaleRotate(shift_limit=aug_shift_limit, rotate_limit=aug_rotate_deg, scale_limit=aug_scale_limit, border_mode=cv2.BORDER_CONSTANT, value=0, p=1) if aug else None,
            A.MultiplicativeNoise(elementwise=True, p=1) if aug else None,
            A.Resize(height=frame_size, width=frame_size, p=1)
            # A.CenterCrop(height=112, width=112, p=1),
        ],
        additional_targets=additional_targets
    )
    return transform


def apply_transform(transform, X):
    """
    Input X have shape [num_frame, x, y] and values between 0-255
    Individual frames to be fed to "transform" should have shape [x, y, c]
    """
    # Make the video into shape [num_frame, x, y, 1] and 'uint8'
    X = np.expand_dims(X, axis=3)
    X = X.astype('uint8')

    # Pack frames of video into a dictionary kwargs
    num_frame = len(transform.additional_targets) + 1
    K = ['image'] + ['image' + str(i) for i in range(1, num_frame, 1)]
    V = [X[0, ...]] + [X[i, ...] for i in range(1, num_frame, 1)]
    kwargs = {key: value for (key, value) in zip(K, V)}

    # Apply transform and stack transformed frames back to a video
    transformed = transform(**kwargs)
    transformed = [transformed['image']] + [transformed['image' + str(i)] for i in range(1, num_frame, 1)]
    transformed = np.stack(transformed, axis=0)

    # Move the channel dimension -> [1, num_frame, x, y]
    X = np.moveaxis(transformed,-1,0)
    return X


def crop_excess(X):
    """
    This is CAMUS-specific and not a general utils function, ensuring the output image has a maximum height of 1232.
    It crops the excess parts of 10 really large images. Other images remain unchanged.
    """
    height = X.shape[1]
    if height == 1945:      # This is special and only applied in patient0038
        X = X[:,:1232,:]
    elif height > 1232:     # This is only for 9 other cases that has really large image size
        # For even number the upper and lower limit would be equal, but for odd number this would be required
        upper_limit =                   int((height - 1232)/2)
        lower_limit = (height - 1232) - int((height - 1232)/2)
        X = X[:,upper_limit:-lower_limit,:]
    return X


class loader(Dataset):
    def __init__(self, data_folder, sub_list, info, aug=False,
                 binarize=False, loader_outputX_channel=1, **kwargs):
        self.data_folder = data_folder
        self.sub_list = sub_list
        self.info = info
        self.aug = aug
        self.binarize = binarize
        self.transform = create_transform(aug, **kwargs)
        self.loader_outputX_channel = loader_outputX_channel

        full_data = []
        for i in range(len(sub_list)):
            ID = sub_list[i]
            X_2view = []
            for view in ['_4CH', '_2CH']:
                X = []
                for phase in ['_ED', '_ES']:        # For each view, load ED & ES to form a 2-channel image
                    filepath = self.data_folder + ID +'/'+ ID + view + phase + '.mhd'
                    frame = io.imread(filepath, plugin='simpleitk')     # frame: [1, x, y]
                    X. append(frame)
                X_2view.append(X)
            full_data.append(X_2view)
        self.full_data = full_data

    def __getitem__(self, index):
        """
        return: X_2view of shape [B, 1, 4, 112, 112] containing 4 frames [A4C ED, A4C ES, A2C ED, A2C ES]
        """
        ID = self.sub_list[index]

        idx = self.info[:,0]==ID
        EDV = self.info[idx, 3].astype(np.float)
        ESV = self.info[idx, 4].astype(np.float)
        EF = self.info[idx, 5].astype(np.float)
        if self.binarize: EF = EF > 53.3
        EDV = torch.from_numpy(EDV).float()
        ESV = torch.from_numpy(ESV).float()
        Y = torch.from_numpy(EF).float() * 0.01

        X_2view = []
        for view in range(2):
            X = []
            for phase in range(2):        # For each view, load ED & ES to form a 2-channel image
                frame = self.full_data[index][view][phase]          # frame: [1, x, y]
                X. append(frame)
            X = np.concatenate(X, axis=0)                           # X: [num_frame=2, x, y]
            # Two images from the same view, with a maximum height 1232, will undergo the same augmentation together.

            X = apply_transform(self.transform, X)                  # X: [c=1, num_frame=2, x, y]
            X_2view.append(X)

        # The final output tensor
        X_2view = np.concatenate(X_2view, axis=1)                   # X_2view: [c=1, num_frame=4, x, y]
        if self.loader_outputX_channel == 0:
            X_2view = X_2view[0,:,:,:]
        X_2view = X_2view / 255
        X_2view = torch.from_numpy(X_2view).float()

        return X_2view, Y, EDV, ESV, ID

    def __len__(self):
        return len(self.sub_list)


def create_dataloader(mode, batch_size=32, data_folder='../data/CAMUS/',
                      split_folder='5fold/',
                      foldk=1,
                      info_dir='../data/CAMUS/info.csv', **kwargs):
    info = np.genfromtxt(info_dir, dtype='str', delimiter=',')

    if '9' in split_folder:
        num_fold = 9
    elif '5' in split_folder:
        num_fold = 5
    if mode == 'test':
        folds_to_load = [foldk]
    elif mode == 'val':
        folds_to_load = [int((foldk + 1) % num_fold)]
    elif mode == 'train':
        folds_to_load = list(range(num_fold))
        folds_to_load.remove(foldk)
        folds_to_load.remove(int((foldk + 1) % num_fold))

    file_list = []
    for k in folds_to_load:
        file_list = file_list + np.load(data_folder + split_folder + '/fold' + str(k) + '.npy').tolist()

    if mode == 'train':
        data = loader(data_folder, file_list, info, aug=True, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    elif (mode == 'val') | (mode == 'test'):
        data = loader(data_folder, file_list, info, aug=False, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
    return dataloader
