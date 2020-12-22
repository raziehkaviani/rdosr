import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import densenet
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Classifier F Training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (defaul)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--dataset', required=True, help='name of the used dataset')
parser.add_argument('--model-dir', dest='model_dir',
                    help='The directory used to load the trained models',
                    default='results_F/modelF_CIFAR10.tar', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the zf corresponding to input',
                    default='results_F', type=str)
parser.add_argument('--unk', nargs='+', required=True, help='label of the unknown class')


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # take unknown classes
    unk_classes = set(list(map(int, args.unk)))
    print('-- Unknown classes are: ', unk_classes)

    # define the classifier
    model = densenet.DenseNet121()
    model = torch.nn.DataParallel(model)

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # load model
    try:
        model.load_state_dict(torch.load(os.path.join(args.model_dir))['state_dict'])
        print("=> loaded model '{}'".format(args.model_dir))
    except:
        raise Exception('''Failed to load the model "%s"''' % args.model_dir)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # load training data
    transformed_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    
    # extract known classes in trainig set
    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=1, shuffle=False)
    print('-- #samples in training set: ', len(train_loader.dataset))

    # load validation data
    transformed_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                                          transforms.ToTensor(), normalize]))
    
    # extract known classes in the validation set
    val_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=1, shuffle=False)
    print('-- #samples in validation set: ', len(val_loader.dataset))

    all_train_zf = []
    all_train_label = []

    # testing
    model.eval()
    for input, target in train_loader:
        if args.cpu == False:
            input = input.cuda(async=True) 

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            zf = model(input)

        all_train_zf.append(zf.data.cpu().numpy())
        target = target.data.numpy()[0]
        if target in unk_classes:
            all_train_label.append(-1)
        else:
            all_train_label.append(target)

    sio.savemat(os.path.join(args.save_dir, 'F_results.mat'), {'train_zf':all_train_zf, 'train_label':all_train_label})

    # make the format of the result to be appropriate for EDC block
    zfs = np.array(all_train_zf)
    n, _, c = np.shape(zfs)
    zfs = zfs.reshape((n, c))
    labels = np.array(all_train_label).reshape((n, 1))

    indices_kwn = np.where(labels!=-1)
    indices_unk = np.where(labels==-1)
     
    gt_kwn = labels[indices_kwn[0],:]+1
    gt_unk = labels[indices_unk[0],:]+2
    zf_kwn = zfs[indices_kwn[0],:]
    zf_unk = zfs[indices_unk[0],:]

    sio.savemat(os.path.join(args.save_dir, args.dataset + '_kwn.mat'), {'x':zf_kwn, 'y':gt_kwn})
    sio.savemat(os.path.join(args.save_dir, args.dataset + '_unk.mat'), {'x':zf_unk, 'y':gt_unk})
   

if __name__ == '__main__':
    main()

