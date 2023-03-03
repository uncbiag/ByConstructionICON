import footsteps
import icon_registration as icon
import torch
import glob
import numpy as np
import nibabel as nib
import itertools
from torch.utils.data import DataLoader, Dataset

import train_knee

BATCH_SIZE = 4
GPUs = 4

#############################################################################
# The following chunck of code is from https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks
# so that we can make fair comparison.
class Dataset_epoch_crop(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
        'Initialization'
        super(Dataset_epoch_crop, self).__init__()
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D_with_crop(self.index_pair[step][0], cropx=160, cropy=144, cropz=192)
        img_B = load_4D_with_crop(self.index_pair[step][1], cropx=160, cropy=144, cropz=192)
        # img_A = zoom(img_A, (1, 0.5, 0.5, 0.5), order=0)
        # img_B = zoom(img_B, (1, 0.5, 0.5, 0.5), order=0)

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()

def load_4D_with_crop(name, cropx, cropy, cropz):
    X = nib.load(name)
    X = X.get_fdata()

    x, y, z = X.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2

    X = X[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]

    X = np.reshape(X, (1,) + X.shape)
    return X

def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm
#############################################################################

class Batch():
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
    
    def make_batch(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter) 
        
        return [d.cuda() for d in data]

if __name__ == "__main__":
    input_shape = [1, 1, 160, 144, 192]
    footsteps.initialize()

    names = sorted(glob.glob('/playpen-ssd/lin.tian/data_local/oasis/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[0:255]
    dataset = DataLoader(Dataset_epoch_crop(names, norm=True), batch_size=BATCH_SIZE*GPUs,
                                         shuffle=True, num_workers=2, drop_last=True)


    loss = train_knee.make_net(input_shape=input_shape)

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, Batch(dataset).make_batch, unwrapped_net=loss)
