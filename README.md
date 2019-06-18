# Capsule-Forensics

Implementation of the paper:  <a href="https://arxiv.org/abs/1906.06876">Multi-task Learning for Detecting and Segmenting Manipulated Facial Images and Videos</a> (BTAS 2019).

You can clone this repository into your favorite directory:

    $ git clone https://github.com/nii-yamagishilab/ClassNSeg

## Requirement
- PyTorch 1.0
- TorchVision
- scikit-learn
- Numpy
- tqdm
- PIL

## Project organization
- Datasets folder, where you can place your training, evaluation, and test set:

      ./datasets
- Checkpoint folder, where the training outputs will be stored:

      ./checkpoints

- Test image folder, where images are used for segmentation demonstration during training:

      ./test_img

Pre-trained models with settings described in our paper are provided in the checkpoints folder.

## Dataset
Each dataset has two parts:
- Original images: ./datasets/\<name\>/\<train;test;validation\>/original
- Altered images: ./datasets/\<name\>/\<train;test;validation\>/altered

All datasets need to be pre-processed to crop facial areas and add segmentation maps. It could be done by using these scripts:

      ./create_dataset_Face2Face.py
      ./create_dataset_Deepfakes.py
      ./create_dataset_FaceSwap.py
**Note**: Parameters with detail explanation could be found in the corresponding source code.

## Training
**Note**: Parameters with detail explanation could be found in the corresponding source code.

    $ python train.py --dataset datasets/face2face/source-to-target --train_set train --val_set validation --outf checkpoints/full --batchSize 64 --niter 100

## Finetuning
Before doing finetuning, copy the best encoder_x.pt and decoder_x.pt checkpoints to checkpoints/finetune with x is the checkpoint number and rename them to encoder_0.pt and decoder_0.pt.
**Note**: Parameters with detail explanation could be found in the corresponding source code.

    $ python finetune.py --dataset datasets/finetune --train_set train --val_set validation --outf checkpoints/finetune --batchSize 64 --niter 50

## Evaluating
**Note**: Parameters with detail explanation could be found in the corresponding source code.
###Classification:

    $ python test_cls.py --dataset <your test dataset> --test_set test --outf checkpoints --id <your selected id>

###Segmentation:

    $ python test_seg.py --dataset <your test dataset> --test_set test --outf checkpoints --id <your selected id>

Beside testing on still images, the proposed method can be applied on videos. One recommendation is using OpenCV 3.4 with Caffe framework for face detection (Visit <a href="https://arxiv.org/abs/1906.06876">here</a> for more information). Another option is using <a href="http://dlib.net/face_detector.py.html">Dlib</a>.

## Authors
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Fuming Fang (https://researchmap.jp/fang/?lang=english)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

## Acknowledgement
This research was supported by JSPS KAKENHI Grant Number JP16H06302, JP18H04120, and JST CREST Grant Number JPMJCR18A6, Japan.

## Reference
H. H. Nguyen, F. Fang, J. Yamagishi, and I. Echizen, “Capsule-Forensics: Multi-task Learning for Detecting and Segmenting Manipulated Facial Images and Videos,” Proc. of the 10th IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS), 8 pages, (September 2019)
