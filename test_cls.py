"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing classification of ClassNSeg (the proposed method)
"""

import os
import torch
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse
from model.ae import Encoder
from model.ae import ActivationLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='datasets/face2face/source-to-target', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='path to test dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--id', type=int, default=46, help="checkpoint ID")
parser.add_argument('--outf', default='checkpoints/full', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    text_writer = open(os.path.join(opt.outf, 'classification.txt'), 'w')

    encoder = Encoder(3)
    act_loss_fn = ActivationLoss()
    
    encoder.load_state_dict(torch.load(os.path.join(opt.outf,'encoder_' + str(opt.id) + '.pt')))
    encoder.eval()

    if opt.gpu_id >= 0:
        encoder.cuda(opt.gpu_id)
        act_loss_fn.cuda(opt.gpu_id)

    class Normalize_3D(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
                Tensor: Normalized image.
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:        """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor

    transform_tns = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    transform_norm = Normalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_tns)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    loss_act_test = 0.0

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0

    for fft_data, labels_data in tqdm(dataloader_test):

        fft_label = labels_data.numpy().astype(np.float)
        labels_data = labels_data.float()

        rgb = transform_norm(fft_data[:,:,:,0:256])

        if opt.gpu_id >= 0:
            rgb = rgb.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        latent = encoder(rgb).reshape(-1, 2, 64, 16, 16)

        zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)

        loss_act = act_loss_fn(zero, one, labels_data)
        loss_act_data = loss_act.item()

        output_pred = np.zeros((fft_data.shape[0]), dtype=np.float)

        for i in range(fft_data.shape[0]):
            if one[i] >= zero[i]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, fft_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        
        pred_prob = torch.softmax(torch.cat((zero.reshape(zero.shape[0],1), one.reshape(one.shape[0],1)), dim=1), dim=1)
        tol_pred_prob = np.concatenate((tol_pred_prob, pred_prob[:,1].data.cpu().numpy()))

        loss_act_test += loss_act_data
        count += 1

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_act_test /= count

    fpr, tpr, thresholds = roc_curve(tol_label, tol_pred_prob, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print('[Epoch %d] act_loss: %.4f  acc: %.2f   eer: %.2f' % (opt.id, loss_act_test, acc_test*100, eer*100))
    text_writer.write('%d,%.4f,%.2f,%.2f\n'% (opt.id, loss_act_test, acc_test*100, eer*100))

    text_writer.flush()
    text_writer.close()