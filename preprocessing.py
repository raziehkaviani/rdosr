import argparse
import numpy as np
from sklearn import preprocessing
from os.path import join, exists
from os import makedirs
import scipy.io as sio


parser = argparse.ArgumentParser(description='preprocessing HSI data')
parser.add_argument('--dataset', required=True, help='name of the used dataset')
parser.add_argument('--unk', nargs='+', required=True, help='label of the unknown class')
args = parser.parse_args()

# HSI datasets
dataset_dict = {'PaviaU':'paviaU', 'Pavia':'pavia', 'Indian_pines':'indian_pines'}

# load data
data = sio.loadmat('dataset/'+args.dataset+'.mat')[dataset_dict[args.dataset]]
gt = sio.loadmat('dataset/'+args.dataset+'_gt.mat')[dataset_dict[args.dataset]+'_gt']

# reshape data
H,W,L = data.shape
data = data.reshape(-1,L)
gt = gt.reshape(-1,1)

# preprocess data
data = preprocessing.scale(data) / 10

# exclude background
notgt_ind = list(np.where(gt==0)[0])

# collect pixels corresponding to unknown classes
unk_inds=[]
for unk in list(map(int, args.unk)):
    unk_ind = list(np.where(gt == unk)[0])
    unk_inds.extend(unk_ind)
    

# split data to Known and Unknown sets
not_include = []
not_include.extend(notgt_ind)
not_include.extend(unk_inds)
mask = np.ones(gt.shape[0], dtype=bool)
mask[not_include] = False

x_kwn = data[mask]
y_kwn = gt[mask]
x_unk = data[unk_inds]
y_unk = gt[unk_inds]
print('--> Known set: ', x_kwn.shape, y_kwn.shape)
print('--> Unknown set: ',  x_unk.shape, y_unk.shape)


# save preprocessed known/unknown split data
save_dir = 'dataset_prep'
if not exists(save_dir):
    makedirs(save_dir)
sio.savemat(join(save_dir, args.dataset+'_kwn.mat'), {'x':x_kwn, 'y':y_kwn})
sio.savemat(join(save_dir, args.dataset+'_unk.mat'), {'x':x_unk, 'y':y_unk})


