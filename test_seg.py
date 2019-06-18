"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing segmentation of ClassNSeg (the proposed method)
"""

import os
import torch
import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics
import argparse
from model.ae import Encoder
from model.ae import Decoder
from model.ae import ActivationLoss
from model.ae import SegmentationLoss

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

    text_writer = open(os.path.join(opt.outf, 'segmentation.txt'), 'w')

    encoder = Encoder(3)
    decoder = Decoder(3)

    encoder.load_state_dict(torch.load(os.path.join(opt.outf,'encoder_' + str(opt.id) + '.pt')))
    encoder.eval()
    decoder.load_state_dict(torch.load(os.path.join(opt.outf,'decoder_' + str(opt.id) + '.pt')))
    decoder.eval()

    if opt.gpu_id >= 0:
        encoder.cuda(opt.gpu_id)
        decoder.cuda(opt.gpu_id)

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

    accuracy = 0.0
    count = 0

    for fft_data, labels_data in tqdm(dataloader_test):

        rgb = transform_norm(fft_data[:,:,:,0:256])
        mask = fft_data[:,0,:,256:512]
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        mask = mask.long()

        if opt.gpu_id >= 0:
            rgb = rgb.cuda(opt.gpu_id)

        latent = encoder(rgb).reshape(-1, 2, 64, 16, 16)

        zero_abs = torch.abs(latent[:,0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:,1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)

        output_pred = torch.zeros(fft_data.shape[0])

        for i in range(fft_data.shape[0]):
            if one[i] >= zero[i]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        y = torch.eye(2)
        if opt.gpu_id >= 0:
            y = y.cuda(opt.gpu_id)
            output_pred = output_pred.cuda(opt.gpu_id)

        y = y.index_select(dim=0, index=output_pred.long())

        latent = (latent * y[:,:,None, None, None]).reshape(-1, 128, 16, 16)

        seg, rect = decoder(latent)

        mask = mask.cpu().numpy()
        seg = seg[:,1,:,:].detach().cpu().numpy()
        seg[seg >= 0.5] = 1.0
        seg[seg < 0.5] = 0.0
        seg = seg.astype(np.uint8)
        
        for i in range(mask.shape[0]):
            
            accuracy += metrics.accuracy_score(mask[i].reshape(mask[i].size), seg[i].reshape(seg[i].size))
            count += 1

    accuracy /= count

    print('[Epoch %d] Accuracy: %.2f' % (opt.id, accuracy*100))
    text_writer.write('%d,%.2f\n'% (opt.id, accuracy*100))

    text_writer.flush()
    text_writer.close()